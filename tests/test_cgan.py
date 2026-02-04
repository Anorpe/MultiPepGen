import pytest
import numpy as np
from unittest.mock import MagicMock
from multipepgen.models.cgan import ConditionalGAN
from multipepgen.config import LABELS

class DummyGenerator:
    def predict(self, x, **kwargs):
        batch_size = x.shape[0]
        seq_len = 5
        vocab_size = 21
        return np.random.rand(batch_size, seq_len, vocab_size, 1)

class DummyDiscriminator:
    def __call__(self, x):
        return np.random.rand(x.shape[0], 1)
    @property
    def trainable_weights(self):
        return []

@pytest.fixture
def cgan():
    gan = ConditionalGAN(sequence_length=5, vocab_size=21, latent_dim=10, num_classes=len(LABELS))
    gan.generator = DummyGenerator()
    gan.discriminator = DummyDiscriminator()
    return gan

def test_cgan_init(cgan):
    assert cgan.sequence_length == 5
    assert cgan.vocab_size == 21
    assert cgan.latent_dim == 10
    assert cgan.num_classes == len(LABELS)

def test_generate_class(cgan):
    df = cgan.generate_class(num_sequences=2, classes=['microbiano'])
    assert 'sequence' in df.columns
    assert len(df) == 2
    assert all(isinstance(s, str) for s in df['sequence'])

def test_generate_class_random(cgan):
    df = cgan.generate_class_random(num_sequences=3)
    assert 'sequence' in df.columns
    assert len(df) == 3

def test_create_generator_model():
    gan = ConditionalGAN(sequence_length=5, vocab_size=21, latent_dim=10, num_classes=len(LABELS))
    model = gan.create_generator_model()
    assert hasattr(model, 'summary')

def test_create_discriminator_model():
    gan = ConditionalGAN(sequence_length=5, vocab_size=21, latent_dim=10, num_classes=len(LABELS))
    model = gan.create_discriminator_model()
    assert hasattr(model, 'summary')

def test_metrics_property(cgan):
    metrics = cgan.metrics
    assert isinstance(metrics, list)
    assert any(m.name == 'generator_loss' for m in metrics)
    assert any(m.name == 'discriminator_loss' for m in metrics) 