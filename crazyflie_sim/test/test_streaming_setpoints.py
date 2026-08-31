"""Tests for simulated streaming position and velocity commands."""

import math

import cffirmware as firm

from crazyflie_interfaces.msg import Position, VelocityWorld

from crazyflie_sim.crazyflie_server import CrazyflieServer
from crazyflie_sim.crazyflie_sil import CrazyflieSIL
from crazyflie_sim.sim_data_types import State

import pytest


@pytest.fixture
def crazyflie():
    """Create a Crazyflie SIL instance without running a controller."""
    return CrazyflieSIL('cf1', [0.0, 0.0, 0.0], 'none', lambda: 0.0)


@pytest.fixture
def server(crazyflie):
    """Create the callback-facing portion of a simulator server."""
    result = CrazyflieServer.__new__(CrazyflieServer)
    result.cfs = {'cf1': crazyflie}
    return result


@pytest.fixture
def quadrotor_class():
    """Load the NumPy rigid-body backend when its dependency is available."""
    pytest.importorskip('rowan')
    from crazyflie_sim.backend.np import Quadrotor
    return Quadrotor


def step_dynamics(crazyflie, quadrotor, clock, steps, dt=0.0005):
    """Advance firmware control and rigid-body dynamics together."""
    for _ in range(steps):
        action = crazyflie.executeController()
        quadrotor.step(action, dt)
        clock[0] += dt
        crazyflie.setState(quadrotor.state)


def test_position_command_configures_position_control(server, crazyflie):
    """A position message selects absolute position control."""
    msg = Position(x=1.0, y=-2.0, z=0.8, yaw=math.pi / 2.0)

    server._cmd_position_changed(msg, 'cf1')

    assert crazyflie.mode == CrazyflieSIL.MODE_LOW_POSITION
    assert crazyflie.setpoint.mode.x == firm.modeAbs
    assert crazyflie.setpoint.mode.y == firm.modeAbs
    assert crazyflie.setpoint.mode.z == firm.modeAbs
    assert crazyflie.setpoint.mode.yaw == firm.modeAbs
    assert crazyflie.setpoint.position.x == pytest.approx(1.0)
    assert crazyflie.setpoint.position.y == pytest.approx(-2.0)
    assert crazyflie.setpoint.position.z == pytest.approx(0.8)
    assert crazyflie.setpoint.attitude.yaw == pytest.approx(90.0)


def test_velocity_command_configures_velocity_control(server, crazyflie):
    """A velocity message selects world-frame velocity control."""
    msg = VelocityWorld(yaw_rate=12.0)
    msg.vel.x = 0.5
    msg.vel.y = -0.25
    msg.vel.z = 0.1

    server._cmd_velocity_world_changed(msg, 'cf1')

    assert crazyflie.mode == CrazyflieSIL.MODE_LOW_VELOCITY
    assert crazyflie.setpoint.mode.x == firm.modeVelocity
    assert crazyflie.setpoint.mode.y == firm.modeVelocity
    assert crazyflie.setpoint.mode.z == firm.modeVelocity
    assert crazyflie.setpoint.mode.yaw == firm.modeVelocity
    assert crazyflie.setpoint.velocity.x == pytest.approx(0.5)
    assert crazyflie.setpoint.velocity.y == pytest.approx(-0.25)
    assert crazyflie.setpoint.velocity.z == pytest.approx(0.1)
    assert crazyflie.setpoint.attitudeRate.yaw == pytest.approx(12.0)


def test_position_command_drives_firmware_controller():
    """A position command produces motor commands through the PID controller."""
    crazyflie = CrazyflieSIL(
        'cf1', [0.0, 0.0, 0.0], 'pid', lambda: 0.01)

    crazyflie.cmdPosition([0.0, 0.0, 1.0], 0.0)
    action = crazyflie.executeController()

    assert min(action.rpm) > 0.0


def test_velocity_command_drives_firmware_controller():
    """A velocity command produces directional motor commands."""
    crazyflie = CrazyflieSIL(
        'cf1', [0.0, 0.0, 1.0], 'pid', lambda: 0.01)
    crazyflie.setState(State([0.0, 0.0, 1.0]))

    crazyflie.cmdVelocityWorld([0.3, 0.0, 0.0], 0.0)
    action = crazyflie.executeController()

    assert min(action.rpm) > 0.0
    assert max(action.rpm) > min(action.rpm)


def test_position_command_moves_simulated_drone(quadrotor_class):
    """Position control moves the rigid-body model toward its target."""
    clock = [0.0]
    state = State([0.0, 0.0, 0.0])
    crazyflie = CrazyflieSIL(
        'cf1', state.pos, 'pid', lambda: clock[0])
    quadrotor = quadrotor_class(state)

    crazyflie.cmdPosition([0.0, 0.0, 1.0], 0.0)
    step_dynamics(crazyflie, quadrotor, clock, 6000)

    assert quadrotor.state.pos[2] > 0.5


def test_velocity_command_moves_simulated_drone(quadrotor_class):
    """Velocity control moves the rigid-body model at the requested speed."""
    clock = [0.0]
    state = State([0.0, 0.0, 1.0])
    crazyflie = CrazyflieSIL(
        'cf1', state.pos, 'pid', lambda: clock[0])
    quadrotor = quadrotor_class(state)

    crazyflie.cmdVelocityWorld([0.3, 0.0, 0.0], 0.0)
    step_dynamics(crazyflie, quadrotor, clock, 4000)

    assert quadrotor.state.pos[0] > 0.3
    assert quadrotor.state.vel[0] == pytest.approx(0.3, abs=0.05)


@pytest.mark.parametrize('first_command', ['position', 'velocity'])
def test_most_recent_streaming_command_selects_mode(
        server, crazyflie, first_command):
    """The most recently received command selects the active control mode."""
    position = Position(x=1.0, y=2.0, z=0.5, yaw=0.0)
    velocity = VelocityWorld(yaw_rate=0.0)
    velocity.vel.x = 0.4

    if first_command == 'position':
        server._cmd_position_changed(position, 'cf1')
        assert crazyflie.mode == CrazyflieSIL.MODE_LOW_POSITION
        server._cmd_velocity_world_changed(velocity, 'cf1')
        assert crazyflie.mode == CrazyflieSIL.MODE_LOW_VELOCITY
        assert crazyflie.setpoint.mode.x == firm.modeVelocity
        assert crazyflie.setpoint.velocity.x == pytest.approx(0.4)
    else:
        server._cmd_velocity_world_changed(velocity, 'cf1')
        assert crazyflie.mode == CrazyflieSIL.MODE_LOW_VELOCITY
        server._cmd_position_changed(position, 'cf1')
        assert crazyflie.mode == CrazyflieSIL.MODE_LOW_POSITION
        assert crazyflie.setpoint.mode.x == firm.modeAbs
        assert crazyflie.setpoint.position.x == pytest.approx(1.0)
