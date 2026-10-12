"""Numerical invariants and actual learning, not just finite-loss smoke tests."""
import io

import pytest
import torch

from PPOAgent import ActorCritic, PPOAgent


def _network(seed=17):
    torch.manual_seed(seed)
    return ActorCritic(6, 3, hidden=32, single_state_dim=3, frame_stack=2,
                       normalize_values=True)


def test_scale_update_preserves_raw_values_and_actor_exactly():
    net = _network()
    states = torch.randn(40, 6)
    before = net.critic_forward(states)[0].detach()
    logits = net.actor_forward(states)[0].detach()
    for targets in (torch.tensor([100., 200.]), torch.tensor([-10000., 5000.])):
        net.update_value_scale(targets)
        torch.testing.assert_close(net.critic_forward(states)[0], before, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(net.actor_forward(states)[0], logits, atol=0, rtol=0)
    assert net.value_count == 4


def test_value_scale_roundtrip_and_legacy_schema():
    net = _network()
    net.update_value_scale(torch.tensor([50., 150.]))
    stream = io.BytesIO()
    torch.save(net.state_dict(), stream)
    stream.seek(0)
    restored = _network(22)
    restored.load_state_dict(torch.load(stream, weights_only=True))
    state = torch.randn(5, 6)
    torch.testing.assert_close(net.critic_forward(state)[0], restored.critic_forward(state)[0])
    legacy = ActorCritic(6, 3)
    assert "value_scale" not in legacy.state_dict()


@pytest.mark.parametrize("seed", [17, 31, 83])
@pytest.mark.parametrize("scale", [1., 100., 10000.])
def test_critic_learns_state_dependent_large_returns(seed, scale):
    net = _network(seed)
    states = torch.randn(256, 6) * 0.4
    targets = scale * (5. + 2. * states[:, 0] - states[:, 4])
    train, test = slice(0, 192), slice(192, None)
    net.update_value_scale(targets[train])
    actor_before = {name: p.detach().clone() for name, p in net.named_parameters()
                    if name.startswith(("actor_", "pi_head"))}
    optimizer = torch.optim.Adam(net.critic_parameters(), lr=0.003)
    for _ in range(240):
        values = net.critic_forward(states[train])[0]
        loss = PPOAgent._critic_value_loss(values / net.value_scale,
                                           targets[train] / net.value_scale)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    predicted = net.critic_forward(states[test])[0].detach()
    residual = targets[test] - predicted
    ev = 1. - residual.var() / targets[test].var()
    relative_rmse = residual.square().mean().sqrt() / targets[test].square().mean().sqrt()
    assert ev > .95, (seed, scale, float(ev))
    assert relative_rmse < .04, (seed, scale, float(relative_rmse))
    for name, parameter in net.named_parameters():
        if name in actor_before:
            torch.testing.assert_close(parameter, actor_before[name], atol=0, rtol=0)


def test_normalized_ppo_update_keeps_raw_gae_targets():
    torch.manual_seed(43)
    agent = PPOAgent(state_dim=3, action_dim=2, gamma=.9, lam=1.,
                     hidden_dim=32, center_rewards=False, normalize_values=True,
                     device="cpu")
    with torch.no_grad():
        agent.net.v_head.weight.zero_()
        agent.net.v_head.bias.zero_()
    expected, running = [], 0.
    rewards = [100.] * 7 + [500.]
    for reward in reversed(rewards):
        running = reward + .9 * running
        expected.append(running)
    for i, reward in enumerate(rewards):
        action, state, logp, value, features, mask = agent.act([i / 8, 0, 0])
        agent.store(state, action, reward, i == 7, None, logp, value,
                    action_features=features, action_mask=mask)
    metrics = agent.update(epochs=4, batch_size=8, bootstrap_value=0.)
    assert metrics["critic_target_mean"] == pytest.approx(sum(expected)/8, rel=1e-5)
    assert metrics["critic_value_scale"] > 100
    assert agent.net.value_count == 8  # no epoch/minibatch double counting
    assert metrics["critic_optimizer_steps"] == 4


def test_normalized_critic_learns_public_state_and_terminal_context_jointly():
    torch.manual_seed(31)
    net = ActorCritic(6, 3, hidden=32, single_state_dim=3, frame_stack=2,
                      normalize_values=True, privileged_critic=True, critic_state_dim=1)
    states = torch.randn(256, 6) * .4
    context = torch.randn(256, 1) * .4
    targets = 100. * (5. + 2. * states[:, 0] - states[:, 4] + 3. * context[:, 0])
    net.update_value_scale(targets[:192])
    optimizer = torch.optim.Adam(net.critic_parameters(), lr=.003)
    for _ in range(300):
        values = net.critic_forward(states[:192], context[:192])[0]
        loss = PPOAgent._critic_value_loss(values / net.value_scale, targets[:192] / net.value_scale)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    error = targets[192:] - net.critic_forward(states[192:], context[192:])[0].detach()
    assert 1. - error.var() / targets[192:].var() > .95
