#!/usr/bin/env python3
import argparse

import cv2

from mmseg.apis import inference_model, init_model, show_result_pyplot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='config module/path')
    parser.add_argument('--checkpoint', required=True, help='checkpoint file')
    parser.add_argument('--img', required=True, help='rgb image path')
    parser.add_argument('--modal-x', default=None, help='depth/modal path')
    parser.add_argument('--out-file', default=None, help='visualization output')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    model = init_model(args.config, args.checkpoint, device=args.device)
    input_data = {'img': args.img, 'modal_x': args.modal_x}
    result = inference_model(model, input_data)
    vis = show_result_pyplot(
        model,
        cv2.cvtColor(cv2.imread(args.img), cv2.COLOR_BGR2RGB),
        result,
        show=False,
        out_file=args.out_file,
    )
    if args.out_file is None:
        print('inference done')
    else:
        print(f'saved to {args.out_file}')


if __name__ == '__main__':
    main()

