import hashlib
import unittest

import bcrypt


def get_password_hash(password):
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password, hashed_password):
    try:
        if len(hashed_password) == 64:
            return hashlib.sha256(password.encode()).hexdigest() == hashed_password
        return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception:
        return False


class AuthHelperTests(unittest.TestCase):
    def test_bcrypt_password_round_trip(self):
        hashed = get_password_hash("correct horse battery staple")
        self.assertTrue(verify_password("correct horse battery staple", hashed))
        self.assertFalse(verify_password("wrong password", hashed))

    def test_legacy_sha256_password_still_verifies(self):
        hashed = hashlib.sha256("legacy-password".encode()).hexdigest()
        self.assertTrue(verify_password("legacy-password", hashed))
        self.assertFalse(verify_password("other-password", hashed))


if __name__ == "__main__":
    unittest.main(verbosity=2)
