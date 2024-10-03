import torch
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

class RoPENd_Torch(torch.nn.Module):
    """N-dimensional Rotary Positional Embedding."""
    def __init__(self, shape, base=10000):
        super(RoPENd_Torch, self).__init__()
        channel_dims, feature_dim = shape[:-1], shape[-1]
        k_max = feature_dim // (2 * len(channel_dims))
        assert feature_dim % k_max == 0, f'shape[-1] ({feature_dim}) is not divisible by 2 * len(shape[:-1]) ({2 * len(channel_dims)})'
        theta_ks = 1 / (base ** (torch.arange(k_max) / k_max))
        angles = torch.cat([t.unsqueeze(-1) * theta_ks for t in
                            torch.meshgrid([torch.arange(d) for d in channel_dims], indexing='ij')], dim=-1)
        rotations = torch.polar(torch.ones_like(angles), angles)
        self.register_buffer('rotations', rotations)

    def forward(self, x):
        x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        pe_x = self.rotations * x
        return torch.view_as_real(pe_x).flatten(-2)

class RoPENd_JAX(nn.Module):
    """N-dimensional Rotary Positional Embedding."""
    shape: tuple
    base: float = 10000

    def setup(self):
        channel_dims, feature_dim = self.shape[:-1], self.shape[-1]
        k_max = feature_dim // (2 * len(channel_dims))
        assert feature_dim % k_max == 0, f'shape[-1] ({feature_dim}) is not divisible by 2 * len(shape[:-1]) ({2 * len(channel_dims)})'

        theta_ks = 1 / (self.base ** (jnp.arange(k_max) / k_max))
        angles = jnp.concatenate([t[..., None] * theta_ks for t in
                                  jnp.meshgrid(*[jnp.arange(d) for d in channel_dims], indexing='ij')], axis=-1)

        self.rotations_cos = jnp.cos(angles)
        self.rotations_sin = jnp.sin(angles)

    def __call__(self, x):
        *dims, last_dim = x.shape
        x_reshaped = x.reshape(*dims, -1, 2)
        x_re, x_im = x_reshaped[..., 0], x_reshaped[..., 1]
        rot_x_re = x_re * self.rotations_cos - x_im * self.rotations_sin
        rot_x_im = x_re * self.rotations_sin + x_im * self.rotations_cos
        return jnp.stack([rot_x_re, rot_x_im], axis=-1).reshape(x.shape)

def test_rope_nd(shape, base=10000):
    torch_model = RoPENd_Torch(shape, base)
    jax_model = RoPENd_JAX(shape, base)
    jax_params = jax_model.init(jax.random.PRNGKey(0), jnp.zeros(shape))

    torch_input = torch.randn(shape)
    jax_input = jnp.array(torch_input.numpy())

    torch_output = torch_model(torch_input)
    jax_output = jax_model.apply(jax_params, jax_input)

    torch_output_np = torch_output.detach().numpy()
    jax_output_np = np.array(jax_output)

    np.testing.assert_allclose(torch_output_np, jax_output_np, rtol=1e-5, atol=1e-5)
    print(f"Test passed for shape {shape}")

# Run tests
test_rope_nd((2, 3, 4, 12)) 
test_rope_nd((5, 7, 9, 48))  # Larger 4D input
test_rope_nd((3, 4, 5, 6, 24))  # 5D input
test_rope_nd((2, 3, 12))  # 3D input
test_rope_nd((10, 100))  # 2D input

print("All tests passed!")
