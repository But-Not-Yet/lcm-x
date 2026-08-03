from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
RELEASE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release.yml"
RELEASE_VERSION = "0.21.0-rc1"
RELEASE_NOTES = REPO_ROOT / ".github" / "release-notes" / f"v{RELEASE_VERSION}.md"


def test_release_workflow_requires_curated_tag_specific_notes():
    workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")

    assert 'NOTES_FILE=".github/release-notes/${GITHUB_REF_NAME}.md"' in workflow
    assert 'body_path: .github/release-notes/${{ github.ref_name }}.md' in workflow
    assert "generate_release_notes: false" in workflow
    assert "git log --pretty" not in workflow


def test_release_workflow_marks_rc_tags_as_prereleases_not_latest():
    workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")

    assert "prerelease: ${{ contains(github.ref_name, '-') }}" in workflow
    assert "make_latest: ${{ contains(github.ref_name, '-') && 'false' || 'true' }}" in workflow
    assert "draft: false" in workflow


def test_release_candidate_identity_surfaces_are_synchronized():
    manifest = (REPO_ROOT / "plugin.yaml").read_text(encoding="utf-8")
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    operator_guide = (REPO_ROOT / "docs" / "operator-guide.md").read_text(
        encoding="utf-8"
    )
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    bug_report = (
        REPO_ROOT / ".github" / "ISSUE_TEMPLATE" / "bug_report.yml"
    ).read_text(encoding="utf-8")

    assert f"version: {RELEASE_VERSION}" in manifest
    assert f"hermes-lcm v{RELEASE_VERSION} (15 tools)" in readme
    assert f"hermes-lcm v{RELEASE_VERSION} (15 tools)" in operator_guide
    assert f"## v{RELEASE_VERSION} - " in changelog
    assert f"v{RELEASE_VERSION}, main, or commit SHA" in bug_report


def test_release_candidate_notes_cover_only_the_merged_release_scope():
    notes = RELEASE_NOTES.read_text(encoding="utf-8")

    assert notes.startswith(f"# hermes-lcm v{RELEASE_VERSION}\n")
    assert all(f"#{number}" in notes for number in (361, 436, 440, 446, 447))
    assert "## Highlights" in notes
    assert "## Changes" in notes
    assert "## Contributors" in notes
    assert "documented harness and corpus" in notes
    assert len(notes.splitlines()) <= 60
