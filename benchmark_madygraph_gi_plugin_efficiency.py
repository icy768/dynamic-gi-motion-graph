"""Time GI reconstruction from pre-acquired buckets, using validation data."""
from __future__ import annotations
import argparse
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from gi_experiment import choose_device, load_frozen, make_bank, runtime_metadata, verify_assets, write_json
from project_paths import DEFAULT_RUN, PROJECT_ROOT, asset_hashes, project_path
from train_graph_protocol_v2 import IMAGE_SIZE, RATES, make_train_validation
from train_madygraph_gi_plugin import ResidualRefiner, csv_write, dgi_from_bucket, gi_forward, physical_residual


def measure(fn, device, warmup=15, repeats=60):
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            resident = torch.cuda.memory_allocated(device)
        else:
            resident = 0
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            fn()
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            samples.append((time.perf_counter() - start) * 1000)
        peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
    return {'median_ms':float(np.median(samples)), 'p90_ms':float(np.percentile(samples,90)),
            'mean_ms':float(np.mean(samples)), 'resident_cuda_memory_mb':resident/1024**2 if peak is not None else None,
            'peak_cuda_memory_mb':peak/1024**2 if peak is not None else None,
            'additional_peak_cuda_memory_mb':max(0,peak-resident)/1024**2 if peak is not None else None}


def benchmark(refiner_dir, output, device, warmup=15, repeats=60):
    manifest = json.loads((refiner_dir/'data_manifest.json').read_text(encoding='utf-8'))
    if manifest['checkpoint_sha256'] != asset_hashes():
        raise RuntimeError('The refiners use different frozen checkpoints')
    torch.set_num_threads(1)
    _, validation = make_train_validation(PROJECT_ROOT)
    target = next(iter(DataLoader(validation,batch_size=1,shuffle=False)))[0].to(device)
    transformer, motion_net = load_frozen(device)
    models = {}
    for kind in ('cnn','fixed','motion'):
        model = ResidualRefiner(kind).to(device).eval()
        model.load_state_dict(torch.load(refiner_dir/f'best_{kind}_refiner.pt',map_location=device,weights_only=True))
        models[kind] = model
    bank = make_bank(device)
    results = {'runtime':runtime_metadata(device),'batch_size':1,'frames':8,'resolution':[IMAGE_SIZE,IMAGE_SIZE],
               'warmup':warmup,'repeats':repeats,
               'input':'Pre-acquired buckets already on device; simulation, acquisition and host-to-device transfer are excluded',
               'data_source':'First validation sequence; reserved final-test data are not read',
               'peak_memory_scope':'All benchmark models are resident; report absolute allocation and extra allocation during each operation',
               'refiner_directory':refiner_dir.relative_to(PROJECT_ROOT).as_posix(),'rates':{}}
    table = []
    with torch.inference_mode():
        for rate in RATES:
            patterns = bank[rate]
            bucket = gi_forward(target,patterns)
            dgi = dgi_from_bucket(bucket,patterns)
            coarse = transformer(dgi)
            scale = manifest['measurement_residual_train_rms_by_rate'][str(rate)]
            residual = (physical_residual(bucket,coarse,patterns)/scale).clamp(-8,8)
            batch, frames = dgi.shape[:2]
            zero_motion = torch.zeros(batch,frames-1,2,device=device)
            def motion_from_dgi(value):
                pairs = torch.cat((value[:,:-1],value[:,1:]),dim=2)
                return motion_net(pairs.reshape(batch*(frames-1),2,IMAGE_SIZE,IMAGE_SIZE)).reshape(batch,frames-1,2)
            motion = motion_from_dgi(dgi)
            def coarse_full():
                return transformer(dgi_from_bucket(bucket,patterns)).clamp(0,1)
            def full(kind):
                value = dgi_from_bucket(bucket,patterns)
                initial = transformer(value)
                feature = (physical_residual(bucket,initial,patterns)/scale).clamp(-8,8)
                edges = motion_from_dgi(value) if kind == 'motion' else zero_motion
                return models[kind](initial,feature,edges).clamp(0,1)
            stages = {
                'dgi':measure(lambda:dgi_from_bucket(bucket,patterns),device,warmup,repeats),
                'transformer_only':measure(lambda:transformer(dgi),device,warmup,repeats),
                'measurement_residual':measure(lambda:(physical_residual(bucket,coarse,patterns)/scale).clamp(-8,8),device,warmup,repeats),
                'motion_estimator':measure(lambda:motion_from_dgi(dgi),device,warmup,repeats)}
            for kind in models:
                stages[kind+'_refiner_only'] = measure(lambda kind=kind:models[kind](coarse,residual,motion if kind == 'motion' else zero_motion).clamp(0,1),device,warmup,repeats)
            pipelines = {'coarse':measure(coarse_full,device,warmup,repeats)}
            for kind in models:
                pipelines[kind] = measure(lambda kind=kind:full(kind),device,warmup,repeats)
            baseline = pipelines['coarse']['median_ms']
            for kind,timing in pipelines.items():
                table.append({'pattern_count':rate,'method':kind,'end_to_end_median_ms':timing['median_ms'],
                              'additional_pipeline_median_ms':timing['median_ms']-baseline,
                              'refiner_only_median_ms':0 if kind == 'coarse' else stages[kind+'_refiner_only']['median_ms'],
                              'peak_cuda_memory_mb':timing['peak_cuda_memory_mb'],
                              'additional_peak_cuda_memory_mb':timing['additional_peak_cuda_memory_mb']})
            results['rates'][str(rate)] = {'stages':stages,'pipelines':pipelines}
    results['additional_time_definition'] = 'Full-pipeline median minus coarse-pipeline median includes residual processing and, for motion, motion estimation. Pure refiner latency is measured separately.'
    write_json(output,results)
    csv_write(output.with_suffix('.csv'),table)
    import matplotlib
    matplotlib.use('Agg',force=True)
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(figsize=(7,4))
    for index,kind in enumerate(('coarse','cnn','fixed','motion')):
        values = [next(row['end_to_end_median_ms'] for row in table if row['pattern_count']==rate and row['method']==kind) for rate in RATES]
        ax.bar(np.arange(2)+(index-1.5)*.18,values,width=.18,label=kind.title())
    ax.set_xticks([0,1],[str(rate) for rate in RATES])
    ax.set_xlabel('Patterns')
    ax.set_ylabel('End-to-end latency (ms)')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output.with_suffix('.png'),dpi=180)
    plt.close(fig)
    print(f'benchmark saved={output}',flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--refiner-dir',default=str(DEFAULT_RUN/'seed0'))
    parser.add_argument('--output',default=str(DEFAULT_RUN/'efficiency.json'))
    parser.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    parser.add_argument('--warmup',type=int,default=15)
    parser.add_argument('--repeats',type=int,default=60)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        parser.error('invalid timing repeat counts')
    verify_assets()
    benchmark(project_path(args.refiner_dir),project_path(args.output),choose_device(args.device),args.warmup,args.repeats)


if __name__ == '__main__':
    main()
