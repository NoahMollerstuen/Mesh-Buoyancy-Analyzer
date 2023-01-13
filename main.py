import math
import re

import matplotlib
import numpy as np
import trimesh
from trimesh import intersections
from trimesh import transformations
from trimesh import visual
import matplotlib.pyplot as plt
import multiprocessing
import click

GRAVITY = np.array((0, -9.8, 0))


def find_waterline(mesh: trimesh.Trimesh, mass: float, fluid_density, tolerance=1e-6) -> (float, trimesh.Trimesh):
    target_volume = mass / fluid_density
    if target_volume > mesh.volume:
        raise ValueError(f"Mesh is not buoyant. Mass: {mass} kg, Volume: {mesh.volume} m^3")

    min_y = mesh.bounds[0][1]
    max_y = mesh.bounds[1][1]
    y = (min_y + max_y) / 2

    volume = 0
    sliced_mesh = None
    while abs(target_volume - volume) > target_volume * tolerance:
        sliced_mesh = intersections.slice_mesh_plane(mesh, (0, -1, 0), (0, y, 0), cap=True)
        volume = sliced_mesh.volume
        if volume < target_volume:
            min_y = y
        else:
            max_y = y
        y = (min_y + max_y) / 2

    assert sliced_mesh is not None
    return y, sliced_mesh


def find_torque(underwater_mesh: trimesh.Trimesh, fluid_density: float) -> np.array:
    # Assumes COM is at 0, 0, 0
    center_of_buoyancy = underwater_mesh.center_mass
    return np.cross(center_of_buoyancy, underwater_mesh.volume * fluid_density * -GRAVITY)


def find_righting_torque(mesh: trimesh.Trimesh, rotation_axis: np.ndarray, rotation_angle: float, mass: float, fluid_density: float):
    mat = transformations.quaternion_matrix(transformations.quaternion_about_axis(rotation_angle, rotation_axis))
    oriented_mesh = mesh.copy()
    oriented_mesh.apply_transform(mat)
    _, underwater_mesh = find_waterline(oriented_mesh, mass, fluid_density)
    tq = find_torque(underwater_mesh, fluid_density)

    # Project torque onto the axis of rotation
    return np.dot(tq, rotation_axis) / np.linalg.norm(rotation_axis)


def plot_torques(mesh: trimesh.Trimesh, mass: float, fluid_density: float, plot_resolution: int):
    print("Plotting righting torques")
    plot_angles = np.linspace(-math.pi, math.pi, plot_resolution, endpoint=True)
    pitch_torques = np.zeros(plot_angles.size)
    roll_torques = np.zeros(plot_angles.size)

    for i, angle in enumerate(plot_angles):
        pitch_torques[i] = find_righting_torque(mesh, np.array((1, 0, 0)), angle, mass, fluid_density)
        roll_torques[i] = find_righting_torque(mesh, np.array((0, 0, 1)), angle, mass, fluid_density)

    matplotlib.use("tkAgg")
    plot_angles_degrees = plot_angles * 180 / math.pi
    fig, (roll_ax, pitch_ax,) = plt.subplots(ncols=2)
    roll_ax.plot(plot_angles_degrees, roll_torques)
    pitch_ax.plot(plot_angles_degrees, pitch_torques)

    for ax, name in ((roll_ax, "Roll"), (pitch_ax, "Pitch")):
        ax.set_title(f"Righting Torque vs {name} Angle", va='bottom')
        ax.set_xlabel(f"{name} Angle (°)")
        ax.set_ylabel(f"{name} Torque (Nm)")
        ax.grid(True)

    plt.show()


def find_and_display_stable_orientation(mesh: trimesh.Trimesh, mass: float, fluid_density: float):
    print("Finding stable orientation")
    rotation = np.array((1, 0, 0, 0))  # Quaternion corresponding to no rotation
    rotational_inertia = mass * math.pow(mesh.volume, 2 / 3)  # For scaling step size with mesh size

    iterations = 0
    while True:
        rotation_matrix = transformations.quaternion_matrix(rotation)

        rotated_mesh = mesh.copy()
        rotated_mesh.apply_transform(rotation_matrix)
        waterline, sliced_mesh = find_waterline(rotated_mesh, mass, fluid_density)
        torque = find_torque(sliced_mesh, fluid_density)
        torque[1] = 0  # There should never be any yaw torque
        step = min(np.linalg.norm(torque) / rotational_inertia / 10, 0.1)

        if iterations > 50 and step < 5e-5:
            break

        rotation = transformations.quaternion_multiply(rotation, transformations.quaternion_about_axis(step, torque))
        iterations += 1

    print("Found stable orientation")
    euler_angles = transformations.euler_from_quaternion(rotation, 'sxzy')
    print(f"Pitch: {math.degrees(euler_angles[0]):.0f}°, Roll: {math.degrees(euler_angles[1]):.0f}°")

    rotated_mesh = mesh.copy()
    rotated_mesh.apply_transform(transformations.quaternion_matrix(rotation))
    waterline, sliced_mesh = find_waterline(rotated_mesh, mass, fluid_density)
    print(f"Draft: {sliced_mesh.extents[1]:.3f} m")

    wetted_area = intersections.slice_mesh_plane(rotated_mesh, (0, -1, 0), (0, waterline, 0), cap=False).area
    print(f"Wetted Area: {wetted_area:.3f} m^2")

    stable_scene = trimesh.scene.Scene()
    stable_scene.add_geometry(rotated_mesh)
    sz = max(mesh.extents[0], mesh.extents[2])
    water_color = (0, 0.25, 0.75, 0.4)
    stable_scene.add_geometry(trimesh.Trimesh(
        vertices=((-sz, waterline, -sz), (-sz, waterline, sz), (sz, waterline, sz), (sz, waterline, -sz)),
        faces=((0, 1, 3), (0, 3, 1), (1, 2, 3), (1, 3, 2)),
        visual=visual.ColorVisuals(face_colors=[water_color] * 4)
    ))

    stable_scene.show()


@click.command()
@click.option("-f", "--filepath", type=str, help="A mesh file to analyze")
@click.option("-m", "--mass", type=float, help="The mass of the body, in kg")
@click.option("-c", "--com", type=str,
              help="The center of mass of the body. Three numbers separated by commas or spaces")
@click.option("-d", "--fluid-density", type=float, default=997, help="The density of the surrounding fluid in kg / m^3")
@click.option("-r", "--plot-resolution", type=int, default=90, help="The number of points on each torque plot")
def main(filepath, mass, com, fluid_density, plot_resolution):
    if filepath is None:
        from tkinter.filedialog import askopenfilename
        filepath = askopenfilename(filetypes=[("Mesh files", " ".join(["." + ext for ext in trimesh.exchange.load.mesh_formats()]))])
        if filepath == "":
            print("No file selected, exiting")
            return

    if mass is None:
        while True:
            print("Please enter the mass of the body in kg")
            mass_raw = input(">")
            try:
                mass = float(mass_raw)
                break
            except ValueError:
                print(f"Could not interpret \"{mass_raw}\" as number")

    need_com_input = com is None
    if not need_com_input:
        com_numbers = re.findall(r"-?[\d|.]+", com)
        if len(com_numbers) == 3:
            com = [float(n) for n in com_numbers]
        else:
            print(f"Could not interpret \"{com}\" as a center of mass")
            need_com_input = True

    if need_com_input:
        com = []
        for axis in ('x', 'y', 'z'):
            while True:
                print(f"Please enter the center of mass {axis} coordinate, in meters")
                pos_raw = input(">")
                try:
                    com.append(float(pos_raw))
                    break
                except ValueError:
                    print(f"Could not interpret \"{pos_raw}\" as number")

    com = np.array(com)

    user_mesh = trimesh.load(filepath)
    user_mesh.apply_scale(0.001)  # Convert from mm to m
    user_mesh.vertices -= com  # Translate the COM to 0, 0, 0

    # Plot torques and find the stable orientations in separate processes
    plot_process = multiprocessing.Process(target=plot_torques, args=(user_mesh, mass, fluid_density, plot_resolution))
    plot_process.start()

    find_and_display_stable_orientation(user_mesh, mass, fluid_density)

    plot_process.join()


if __name__ == "__main__":
    main()
