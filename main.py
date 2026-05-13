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
                        choices=['raw', 'fm', 'fm_ft', 'fm_query', 'fm_hcls',
                                 'fm_e3', 'fm_query_e3',
                                 'fm_decoder', 'fm_decoder_ft',
                                 'fm_decoder2', 'fm_decoder2_ft',
                                 'all', 'all_e3', 'all_decoder',
                                 'decoder2', 'thesis', 'thesis2', 'de'])
    parser.add_argument('--lr',           type=float, default=1e-3)
    parser.add_argument('--lr_raw',       type=float, default=1e-4)
    parser.add_argument('--lr_ft',        type=float, default=1e-4)
    parser.add_argument('--max_patients',       type=int, default=None)
    parser.add_argument('--max_train_patients', type=int, default=None,
                        help='Cap only the training split for data efficiency runs')
    parser.add_argument('--batch_size',   type=int,   default=128)
    parser.add_argument('--eval_only',    action='store_true',
                        help='Skip training; load saved weights and run evaluation only')
    parser.add_argument('--bedtime_only', action='store_true',
                        help='Hypo risk: bedtime filter (20:00–23:59) + 8h horizon')
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
