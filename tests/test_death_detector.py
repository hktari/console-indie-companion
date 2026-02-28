from src.context.detectors import DeathDetector


def test_death_detector():
    detector = DeathDetector()

    # 1. Normal fighting
    scene1 = {
        "location": "Dark Tomb",
        "activity": "fighting",
        "enemies": ["Disquiet Being"],
    }
    assert detector.detect(scene1) is None
    assert detector._is_dying is False

    # 2. Death occurs (location: null, activity: fighting)
    scene2 = {"location": None, "activity": "fighting", "enemies": ["voidling"]}
    event = detector.detect(scene2)
    assert event == "[SYSTEM EVENT] The player has died."
    assert detector._is_dying is True

    # 3. Still in death state (location: null, activity: fighting) - should not trigger again
    scene3 = {"location": None, "activity": "fighting", "enemies": ["none"]}
    assert detector.detect(scene3) is None
    assert detector._is_dying is True

    # 4. Respawn (location: valid, activity: exploring)
    scene4 = {
        "location": "Rooted Ziggurat",
        "activity": "exploring",
        "enemies": ["none"],
    }
    assert detector.detect(scene4) is None
    assert detector._is_dying is False

    print("DeathDetector tests passed!")


if __name__ == "__main__":
    test_death_detector()
