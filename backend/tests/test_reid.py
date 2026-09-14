import uuid
import numpy as np
import pytest
from reid_store import reid_store


@pytest.fixture(autouse=True)
def clean_store():
    reid_store.clear()
    yield
    reid_store.clear()


def test_cross_camera_linking():
    """Verify that similar embeddings from DIFFERENT cameras get linked
    under the same global_person_id."""
    
    # Generate a base embedding
    base_emb = np.random.randn(512).astype(np.float32)
    base_emb /= np.linalg.norm(base_emb) + 1e-8

    # Slight perturbation for the second sighting (to simulate real-world noise)
    noise = np.random.randn(512).astype(np.float32) * 0.05
    emb2 = base_emb + noise
    emb2 /= np.linalg.norm(emb2) + 1e-8

    # Add first sighting (Camera A)
    s1 = reid_store.add_sighting(
        camera_id="cam-A",
        track_id=1,
        embedding=base_emb,
        similarity_threshold=0.5,
    )
    assert s1.global_person_id is None, "First sighting shouldn't be linked to anything yet"

    # Add second sighting (Camera B) - highly similar
    s2 = reid_store.add_sighting(
        camera_id="cam-B",
        track_id=1,
        embedding=emb2,
        similarity_threshold=0.5,
    )

    # They should now be linked
    assert s2.global_person_id is not None
    
    # Reload s1 from store to check if it was retroactively linked
    s1_updated = reid_store._sightings[s1.sighting_id]
    assert s1_updated.global_person_id == s2.global_person_id

    # Check the query API
    persons = reid_store.get_global_persons()
    assert len(persons) == 1
    p = persons[0]
    assert p["global_person_id"] == s2.global_person_id
    assert len(p["sightings"]) == 2
    assert set(p["cameras"]) == {"cam-A", "cam-B"}


def test_different_people_not_linked():
    """Verify that dissimilar embeddings do not get linked."""
    emb1 = np.random.randn(512).astype(np.float32)
    emb2 = np.random.randn(512).astype(np.float32)

    s1 = reid_store.add_sighting(
        camera_id="cam-A",
        track_id=1,
        embedding=emb1,
        similarity_threshold=0.5,
    )
    s2 = reid_store.add_sighting(
        camera_id="cam-B",
        track_id=2,
        embedding=emb2,
        similarity_threshold=0.5,  # with random 512-d, similarity is near 0
    )

    assert s1.global_person_id is None
    assert s2.global_person_id is None
    assert len(reid_store.get_global_persons()) == 0


def test_same_camera_not_linked():
    """Verify that sightings from the SAME camera are not linked to each other
    by the cross-camera matcher."""
    base_emb = np.random.randn(512).astype(np.float32)
    
    s1 = reid_store.add_sighting(
        camera_id="cam-A",
        track_id=1,
        embedding=base_emb,
        similarity_threshold=0.5,
    )
    s2 = reid_store.add_sighting(
        camera_id="cam-A",
        track_id=2,
        embedding=base_emb,
        similarity_threshold=0.5,
    )

    assert s1.global_person_id is None
    assert s2.global_person_id is None
    assert len(reid_store.get_global_persons()) == 0


def test_identity_name_propagation():
    """Verify that if one sighting has a known identity name, it propagates
    to the global person record."""
    emb = np.random.randn(512).astype(np.float32)

    # First sighting is anonymous
    s1 = reid_store.add_sighting(
        camera_id="cam-A",
        track_id=1,
        embedding=emb,
        similarity_threshold=0.5,
    )

    # Second sighting is matched by video_gallery as "Alice"
    s2 = reid_store.add_sighting(
        camera_id="cam-B",
        track_id=1,
        embedding=emb,
        similarity_threshold=0.5,
        identity_name="Alice",
    )

    # The global person should now be named "Alice"
    persons = reid_store.get_global_persons()
    assert len(persons) == 1
    assert persons[0]["identity_name"] == "Alice"
    
    # Both individual sightings should have the name
    assert persons[0]["sightings"][0]["identity_name"] == "Alice"
    assert persons[0]["sightings"][1]["identity_name"] == "Alice"
