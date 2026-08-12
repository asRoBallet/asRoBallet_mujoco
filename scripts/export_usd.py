import argparse
from collections import Counter
from pathlib import Path
import shutil

import mujoco_usd_converter
from pxr import Usd


DEFAULT_INPUT_PATH = Path("robots/mjcf/asRoBallet.xml")
DEFAULT_OUTPUT_DIR = Path("robots/usd")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert an asRoBallet MJCF asset to OpenUSD."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Input MJCF path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output Atomic Component directory.",
    )
    parser.add_argument(
        "--single-layer",
        action="store_true",
        help="Write one USD layer instead of an Atomic Component structure.",
    )
    parser.add_argument(
        "--no-physics-scene",
        action="store_true",
        help="Do not author the UsdPhysics.Scene prim.",
    )
    parser.add_argument(
        "--comment",
        default="Exported from asRoBallet MJCF.",
        help="Comment stored in the USD asset.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output directory.",
    )
    return parser.parse_args()


def prepare_output_directory(output_dir, force):
    if not output_dir.exists():
        return
    if not output_dir.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {output_dir}")
    if not any(output_dir.iterdir()):
        return
    if not force:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use --force to replace it."
        )

    resolved = output_dir.resolve()
    protected = {Path.cwd().resolve(), Path.home().resolve(), Path("/")}
    if resolved in protected:
        raise ValueError(f"Refusing to replace protected directory: {resolved}")
    shutil.rmtree(output_dir)


def convert_mjcf(
    input_path,
    output_dir,
    layer_structure=True,
    physics_scene=True,
    comment="",
    force=False,
):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if not input_path.is_file():
        raise FileNotFoundError(f"MJCF file not found: {input_path}")
    prepare_output_directory(output_dir, force)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    converter = mujoco_usd_converter.Converter(
        layer_structure=layer_structure,
        scene=physics_scene,
        comment=comment,
    )
    asset = converter.convert(str(input_path), str(output_dir))
    asset_path = Path(asset.path)

    stage = Usd.Stage.Open(str(asset_path))
    if stage is None or not stage.GetDefaultPrim().IsValid():
        raise RuntimeError(f"Converted USD asset is invalid: {asset_path}")
    prim_types = Counter(prim.GetTypeName() for prim in stage.Traverse())
    return asset_path, prim_types


def main():
    args = parse_args()
    asset_path, prim_types = convert_mjcf(
        args.input_path,
        args.output_dir,
        layer_structure=not args.single_layer,
        physics_scene=not args.no_physics_scene,
        comment=args.comment,
        force=args.force,
    )
    print(f"Exported: {asset_path}")
    print(
        f"meshes={prim_types['Mesh']} "
        f"revolute_joints={prim_types['PhysicsRevoluteJoint']} "
        f"actuators={prim_types['MjcActuator']}"
    )


if __name__ == "__main__":
    main()
