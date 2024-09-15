from os import path
from argparse import ArgumentParser

import engine
from drs.data import BaseData
from drs.loggers import LocalLogger, WandbLogger

parser = ArgumentParser('NLOS LEAP training script')
parser.add_argument('config', type=str, help='Path to the config file')
parser.add_argument('--name', '-n', type=str, default=None, help='Name of the experiment')
parser.add_argument('--debug', '-d', action='store_true', default=False, help='debug mode (for sanity check)')
parser.add_argument('--gpus', '-g', default=0, type=int, help='GPU id to use.')


def main():
    args = parser.parse_args()
    device = f'cuda:{args.gpus}'
    cfg = engine.load_config(args.config)
    engine.create_experiment_context(cfg.get('output_dir', 'runs'), args.name)

    data: BaseData = engine.instantiate(cfg.data)
    data.preprocess()
    data.prepare()
    data = data.to(device)
    print('data loaded.')

    loggers = [LocalLogger(save_dir=engine.to_experiment_dir())]
    wandb_config_path = cfg.get('wandb_config_path', None)
    if (wandb_config_path is not None) and path.exists(wandb_config_path) and not args.debug:
        loggers.append(WandbLogger(name=args.name,
                                   save_dir=engine.to_experiment_dir('wandb'),
                                   config_path=wandb_config_path))
    for logger in loggers:
        logger.init()
        logger.add_config(cfg)

    init_resolution = (data.Z, data.N)
    grid = engine.instantiate(cfg.grid, init_resolution=init_resolution)
    sampler = engine.instantiate(cfg.sampler, hidden_size=data.wall_size)
    solver = engine.instantiate(cfg.solver, grid=grid, sampler=sampler, data=data, loggers=loggers)
    print('components loaded.')

    if args.debug:
        solver.enable_debug_mode()
    solver = solver.to(device)
    solver.run()


if __name__ == '__main__':
    main()
