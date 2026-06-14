from robocasa.environments.kitchen.kitchen import *

_SUPPORTED_MOVE_FIXTURES = {
    "counter": FixtureType.COUNTER,
    "sink": FixtureType.SINK,
    "stove": FixtureType.STOVE,
    "cabinet": FixtureType.CABINET,
    "microwave": FixtureType.MICROWAVE,
    "fridge": FixtureType.FRIDGE,
}


def resolve_move_fixture_type(fixture):
    """Resolve fixture id from FixtureType, int, or string name."""
    if isinstance(fixture, FixtureType):
        return fixture
    if isinstance(fixture, int):
        return FixtureType(fixture)
    key = str(fixture).lower().strip()
    if key not in _SUPPORTED_MOVE_FIXTURES:
        supported = ", ".join(sorted(_SUPPORTED_MOVE_FIXTURES))
        raise ValueError(
            f"Unsupported fixture '{fixture}'. Supported names: {supported}"
        )
    return _SUPPORTED_MOVE_FIXTURES[key]


def _fixture_label(fixture):
    return type(fixture).__name__.lower()


def _any_fixture_type(fixture, fixture_types):
    return any(fixture_is_type(fixture, ft) for ft in fixture_types)


class MovePan(Kitchen):
    """
    Move a pan from one fixture to another.

    Args:
        source_fixture: Initial pan placement fixture (counter, sink, stove, etc.).
        target_fixture: Success is checked when the pan reaches this fixture.
    """

    EXCLUDE_LAYOUTS = [8]

    def __init__(
        self,
        source_fixture=FixtureType.COUNTER,
        target_fixture=FixtureType.SINK,
        *args,
        **kwargs,
    ):
        self._source_fixture_id = resolve_move_fixture_type(source_fixture)
        self._target_fixture_id = resolve_move_fixture_type(target_fixture)
        if self._source_fixture_id == self._target_fixture_id:
            raise ValueError("source_fixture and target_fixture must differ")
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.source = self.register_fixture_ref(
            "source", dict(id=self._source_fixture_id)
        )
        self.target = self.register_fixture_ref(
            "target", dict(id=self._target_fixture_id)
        )

        if fixture_is_type(self.source, FixtureType.COUNTER):
            self.counter = self.source
        elif fixture_is_type(self.target, FixtureType.COUNTER):
            self.counter = self.target
        else:
            self.counter = self.register_fixture_ref(
                "counter",
                dict(id=FixtureType.COUNTER, ref=self.target, size=(0.6, 0.4)),
            )

        if _any_fixture_type(
            self.target, (FixtureType.SINK, FixtureType.STOVE)
        ) or fixture_is_type(self.source, FixtureType.STOVE):
            self.init_robot_base_ref = self.target
        else:
            self.init_robot_base_ref = self.source

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        source_name = _fixture_label(self.source)
        target_name = _fixture_label(self.target)
        prep = "in" if _any_fixture_type(
            self.target,
            (
                FixtureType.SINK,
                FixtureType.CABINET,
                FixtureType.MICROWAVE,
                FixtureType.FRIDGE,
            ),
        ) else "on"
        ep_meta[
            "lang"
        ] = f"Pick the pan from the {source_name} and place it {prep} the {target_name}."
        return ep_meta

    def _get_obj_cfgs(self):
        if fixture_is_type(self.source, FixtureType.STOVE):
            placement = dict(
                fixture=self.source,
                ensure_object_boundary_in_range=False,
                size=(0.20, 0.20),
            )
        elif fixture_is_type(self.source, FixtureType.COUNTER):
            placement = dict(
                fixture=self.source,
                sample_region_kwargs=dict(
                    ref=self.target,
                    loc="left_right",
                    top_size=(0.6, 0.4),
                ),
                size=(0.55, 0.55),
                pos=("ref", -1.0),
                ensure_valid_placement=False,
            )
        elif fixture_is_type(self.source, FixtureType.SINK):
            placement = dict(
                fixture=self.source,
                size=(0.25, 0.25),
                pos=(0.0, 1.0),
            )
        else:
            placement = dict(
                fixture=self.source,
                size=(0.30, 0.30),
            )

        return [
            dict(
                name="pan",
                obj_groups=("pan",),
                graspable=True,
                washable=True,
                max_size=(0.55, 0.55, None),
                placement=placement,
            )
        ]

    def _check_success(self):
        if not OU.gripper_obj_far(self, obj_name="pan"):
            return False

        if fixture_is_type(self.target, FixtureType.SINK):
            return OU.obj_inside_of(self, "pan", self.target, partial_check=True)
        if _any_fixture_type(
            self.target,
            (FixtureType.CABINET, FixtureType.MICROWAVE, FixtureType.FRIDGE),
        ):
            return OU.obj_inside_of(self, "pan", self.target, partial_check=False)
        if _any_fixture_type(self.target, (FixtureType.COUNTER, FixtureType.STOVE)):
            return OU.check_obj_fixture_contact(self, "pan", self.target)
        return False
