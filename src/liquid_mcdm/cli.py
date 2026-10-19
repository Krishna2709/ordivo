import argparse
import json
from pathlib import Path

from liquid_mcdm.data import generate_dataset
from liquid_mcdm.evaluation import evaluate_dataset


def main():
    p = argparse.ArgumentParser(description='CPU MCDM lab; no implicit model or GPU actions')
    commands = p.add_subparsers(dest='command', required=True)
    g = commands.add_parser('generate')
    g.add_argument('--output',type=Path,required=True)
    g.add_argument('--count',type=int,default=1000)
    g.add_argument('--seed',type=int,default=42)
    b = commands.add_parser('baseline')
    b.add_argument('--data',type=Path,required=True)
    b.add_argument('--split',choices=['train','val','test'],default='test')
    b.add_argument('--output',type=Path,required=True)
    b.add_argument('--wandb',choices=['offline','online'])
    b.add_argument('--entity',default=None)
    commands.add_parser('environment')
    t = commands.add_parser('tokenizer-check')
    t.add_argument('--data',type=Path,required=True)
    t.add_argument('--output',type=Path,required=True)
    t.add_argument('--cache',type=Path,default=Path('.cache/d1-tokenizer'))
    args = p.parse_args()
    from liquid_mcdm.runtime import require_remote_gpu
    require_remote_gpu()
    if args.command == 'generate':
        result = generate_dataset(args.output,count=args.count,seed=args.seed)
    elif args.command == 'baseline':
        result = evaluate_dataset(args.data,args.split)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        if args.wandb:
            from liquid_mcdm.tracking import log_cpu_run
            result['tracking'] = log_cpu_run(args.data/'manifest.json',args.output,
                                             args.output.parent/'tracking',mode=args.wandb,
                                             entity=args.entity)
    elif args.command == 'tokenizer-check':
        from liquid_mcdm.training.preflight import tokenizer_preflight
        result = tokenizer_preflight(args.data,args.cache)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
    else:
        from liquid_mcdm.environment import environment_report
        result = environment_report()
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
