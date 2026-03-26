import unittest
from unittest.mock import Mock, patch

def fetch_data(api_client):
    response = api_client.get("/data")
    if response.status_code == 200:
        return response.json()
    return None

class TestFetchData(unittest.TestCase):
    def test_fetch_data_success(self):
        mock_client = Mock()
        mock_client.get.return_value.status_code = 200
        mock_client.get.return_value.json.return_value = {"key": "value"}
        result = fetch_data(mock_client)
        self.assertEqual(result, {"key": "value"})
        mock_client.get.assert_called_once_with("/data")

    @patch('__main__.fetch_data')
    def test_fetch_data_patch(self, mock_fetch):
        mock_fetch.return_value = {"mocked": "data"}
        result = fetch_data(None)
        self.assertEqual(result, {"mocked": "data"})

if __name__ == "__main__":
    unittest.main(verbosity=2)
