import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='TVAE Stage 2 — downstream applications')
    parser.add_argument('--app',          required=True,
                        choices=['imputation', 'forecasting', 'hypo_risk',
                                 'isf_cr', 'digital_twin', 'tir'])
    parser.add_argument('--run_id',       required=True)
    parser.add_argument('--data',         default='data/processed/adults')
    parser.add_argument('--epochs',       type=int,   default=50)
    parser.add_argument('--mode',         default='all',
                        choices=['raw', 'fm', 'ft', 'fm_query', 'fm_hcls', 'all'])
    parser.add_argument('--lr',           type=float, default=1e-3)
    parser.add_argument('--lr_raw',       type=float, default=1e-4)
    parser.add_argument('--lr_ft',        type=float, default=1e-4)
    parser.add_argument('--max_patients', type=int,   default=None)
    parser.add_argument('--batch_size',   type=int,   default=128)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.app == 'imputation':
        from src.stage2.apps.imputation import run
    elif args.app == 'forecasting':
        from src.stage2.apps.forecasting import run
    elif args.app == 'hypo_risk':
        from src.stage2.apps.hypo_risk import run
    elif args.app == 'isf_cr':
        from src.stage2.apps.isf_cr import run
    elif args.app == 'digital_twin':
        from src.stage2.apps.digital_twin import run
    elif args.app == 'tir':
        from src.stage2.apps.tir import run

    run(args)


if __name__ == '__main__':
    main()
