#!/usr/bin/env python3
"""Pure-Jittor checkpoint key inspection utility."""

import jittor as jt


def main():
    checkpoint = jt.load('checkpoints/trained/NYUv2_DFormer_Large.pth')
    state_dict = checkpoint['model'] if isinstance(
        checkpoint, dict) and 'model' in checkpoint else checkpoint

    print('All checkpoint parameters:')
    for key in sorted(state_dict.keys()):
        shape = getattr(state_dict[key], 'shape', None)
        print(f'  {key}: {shape}')

    print(f'\nTotal parameters: {len(state_dict)}')
    print('\nClassification related parameters:')
    for key in sorted(state_dict.keys()):
        if any(word in key.lower()
               for word in ['head', 'pred', 'cls', 'classifier', 'fuse', 'linear']):
            shape = getattr(state_dict[key], 'shape', None)
            print(f'  {key}: {shape}')


if __name__ == '__main__':
    main()

