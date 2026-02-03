"""
Example test to verify pytest infrastructure is working.
"""


def test_taxonomy_fixture_works(mock_taxonomy):
    """Verify mock taxonomy fixture creates correct structure."""
    assert mock_taxonomy.root.name == "Root"
    assert len(mock_taxonomy.root.children) == 2
    
    # Verify child nodes
    child_names = [c.name for c in mock_taxonomy.root.children]
    assert "Tech" in child_names
    assert "Finance" in child_names
