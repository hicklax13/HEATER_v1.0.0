from starlette.testclient import TestClient

from api.contracts.closers import CloserEntry, ClosersResponse
from api.contracts.common import PlayerRef
from api.deps import get_closer_service
from api.main import create_app


def test_to_entry_nan_mlb_id_does_not_raise():
    """A NaN mlb_id must not raise: ``int(mlb_id) if mlb_id is not None`` does NOT
    catch NaN (``nan is not None`` is True) → ``int(nan)`` ValueError."""
    from api.services.closers_service import CloserService

    row = {
        "team": "NYY",
        "closer_name": "Devin Williams",
        "setup_names": "",
        "job_security": 0.8,
        "projected_sv": 30.0,
        "era": 2.50,
        "whip": 1.05,
        "mlb_id": float("nan"),
    }
    entry = CloserService._to_entry(row, None)  # must not raise
    assert entry.team == "NYY"
    assert entry.closer is not None
    assert entry.closer.mlb_id in (None, 0)  # NaN id degrades, never crashes


def test_closers_contract_shape():
    resp = ClosersResponse(
        entries=[
            CloserEntry(
                team="NYY",
                closer=PlayerRef(id=1, name="Clay Holmes", positions="RP"),
                role="Closer",
                confidence="Firm",
                handcuffs=[PlayerRef(id=2, name="Jonathan Loaisiga", positions="RP")],
            )
        ]
    )
    dumped = resp.model_dump()
    assert dumped["entries"][0]["team"] == "NYY"
    assert dumped["entries"][0]["closer"]["name"] == "Clay Holmes"
    assert dumped["entries"][0]["handcuffs"][0]["name"] == "Jonathan Loaisiga"
    # defaults
    assert CloserEntry(team="BOS").closer is None
    assert CloserEntry(team="BOS").handcuffs == []
    assert CloserEntry(team="BOS").role == ""
    assert CloserEntry(team="BOS").confidence == ""


class _FakeCloserService:
    def get_closers(self) -> ClosersResponse:
        return ClosersResponse(
            entries=[
                CloserEntry(
                    team="NYY",
                    closer=PlayerRef(id=1, name="Clay Holmes", positions="RP"),
                    role="Closer",
                    confidence="Firm",
                    handcuffs=[],
                )
            ]
        )


def test_get_closers_returns_contract():
    app = create_app()
    app.dependency_overrides[get_closer_service] = lambda: _FakeCloserService()
    client = TestClient(app)
    resp = client.get("/api/closers")
    assert resp.status_code == 200
    body = resp.json()
    assert body["entries"][0]["team"] == "NYY"
    assert body["entries"][0]["closer"]["name"] == "Clay Holmes"
