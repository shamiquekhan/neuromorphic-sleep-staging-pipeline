"""Tests for model architecture and dataset loading."""

import numpy as np
import pytest
import pandas as pd
import torch
from pathlib import Path

from src.models.improved_student import (
    ImprovedStudent, DepthwiseSeparableConv1d,
    LiteMultiResolutionStem, ParametricGabor1D,
    CompactGaborFEB, count_parameters,
)


class TestModelArchitecture:
    def test_dwsep_conv_shape(self):
        layer = DepthwiseSeparableConv1d(4, 16, kernel=5, stride=1)
        x = torch.randn(1, 4, 3000)
        out = layer(x)
        assert out.shape[1] == 16

    def test_stem_output_channels(self):
        stem = LiteMultiResolutionStem(in_ch=4, width=10, fs=100)
        x = torch.randn(1, 4, 3000)
        out = stem(x)
        assert out.shape[1] == 20  # width * 2

    def test_gabor_output_shape(self):
        gabor = ParametricGabor1D(in_ch=4, n_filters=8)
        x = torch.randn(1, 4, 3000)
        out = gabor(x)
        assert out.shape[1] == 4 * 8  # in_ch * n_filters

    def test_gabor_feb_output(self):
        feb = CompactGaborFEB(in_ch=4, n_filters=8, out_dim=32)
        x = torch.randn(1, 4, 3000)
        out = feb(x)
        assert out.shape == (1, 32)

    def test_full_model_forward(self):
        model = ImprovedStudent(n_classes=5)
        x = torch.randn(2, 10, 4, 3000)
        out = model(x)
        assert out.shape == (2, 10, 5)

    def test_parameter_count_reasonable(self):
        model = ImprovedStudent(n_classes=5)
        n = count_parameters(model)
        assert 40_000 < n < 150_000, f"Expected 40K-150K params, got {n}"

    def test_gru_temporal_modeling(self):
        model = ImprovedStudent(n_classes=5)
        x = torch.randn(1, 10, 4, 3000)
        out1 = model(x)
        x2 = torch.randn(1, 10, 4, 3000)
        out2 = model(x2)
        assert not torch.allclose(out1, out2), "GRU should produce different outputs for different inputs"
