from trainer import Trainer
import argparse
from PIL import Image
import os

parser = argparse.ArgumentParser()
parser.add_argument("--type", default='gan')
parser.add_argument("--lr", default=0.0002, type=float)
parser.add_argument('--lr_lower_boundary', default=2e-6, type=float)
parser.add_argument('--lr_update_type', default=1, type=int)
parser.add_argument('--lr_update_step', default=3000, type=int)
parser.add_argument("--l1_coef", default=50, type=float)
parser.add_argument("--l2_coef", default=100, type=float)
parser.add_argument("--diter", default=5, type=int)
parser.add_argument("--cls", default=False, action='store_true')
parser.add_argument("--project", default=False, action='store_true')
parser.add_argument("--concat", default=False, action='store_true')
parser.add_argument("--vis_screen", default='gan')
parser.add_argument("--save_path", default='')
parser.add_argument("--inference", default=False, action='store_true')
parser.add_argument('--pre_trained_disc', default=None)
parser.add_argument('--pre_trained_gen', default=None)
parser.add_argument('--dataset', default='youtubers')
parser.add_argument('--split', default=None, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--h', default=64, type=int)
parser.add_argument('--scale_size', default=64, type=int)
parser.add_argument('--nc', default=64, type=int)
parser.add_argument('--k', default=0, type=float)
parser.add_argument('--gamma', default=0.5, type=float)
parser.add_argument('--lambda_k', default=0.001, type=float)
args = parser.parse_args()

trainer = Trainer(type=args.type,
                  dataset=args.dataset,
                  split=args.split,
                  lr=args.lr,
                  lr_lower_boundary=args.lr_lower_boundary,
                  lr_update_type=args.lr_update_type,
                  lr_update_step=args.lr_update_step,
                  diter=args.diter,
                  vis_screen=args.vis_screen,
                  save_path=args.save_path,
                  l1_coef=args.l1_coef,
                  l2_coef=args.l2_coef,
                  pre_trained_disc=args.pre_trained_disc,
                  pre_trained_gen=args.pre_trained_gen,
                  batch_size=args.batch_size,
                  num_workers=args.num_workers,
                  epochs=args.epochs,
                  h=args.h,
                  scale_size=args.scale_size,
                  num_channels=args.nc,
                  k=args.k,
                  lambda_k=args.lambda_k,
                  gamma=args.gamma,
                  project=args.project,
                  concat=args.concat,
                  )

if not args.inference:
    trainer.train(args.cls)
else:
    trainer.predict()

