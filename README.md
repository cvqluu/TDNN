# TDNN
Simple Time Delay Neural Network (TDNN) implementation in Pytorch. Uses the unfold method to slide over an input sequence.

![Alt text](misc/diagram.png?raw=true "Diagram") [1] https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf

# Factorized TDNN (TDNN-F)

I've also implemented the Factorized TDNN from Kaldi (TDNN-F) in PyTorch here: https://github.com/cvqluu/Factorized-TDNN

## Usage

To recreate the TDNN part of the x-vector network in [2]:

```python

from tdnn import TDNN

# Assuming 24 dim MFCCs per frame

frame1 = TDNN(input_dim=24, output_dim=512, context_size=5, dilation=1)
frame2 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=2)
frame3 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=3)
frame4 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1)
frame5 = TDNN(input_dim=512, output_dim=1500, context_size=1, dilation=1)

# Input to frame1 is of shape (batch_size, T, 24)
# Output of frame5 will be (batch_size, T-14, 1500)

```

![Alt text](misc/xvec_config.png?raw=true "Diagram") [2] https://www.danielpovey.com/files/2018_icassp_xvectors.pdf
