"""Checks for optional cognition in the shared SEA-AD loader."""
from paper_watch.alztest import data


def test_annotations_only_require_cognition_when_requested(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    (tmp_path / "lec_cells.csv").write_text(
        "cell_label,cell_barcode,barcoded_cell_sample_label,donor_label\n"
        "cell1,barcode1,sample1,donor1\n"
    )
    (tmp_path / "lec_clusters.csv").write_text("cell_label,label\ncell1,cluster1\n")
    (tmp_path / "cluster_terms.csv").write_text("cluster_label,subclass\ncluster1,IT\n")
    disease = tmp_path / "disease.csv"
    header = "donor_label,Overall AD neuropathological Change"
    disease.write_text(f"{header}\ndonor1,High\n")

    basic = data.seaad_annotations()
    assert basic.loc["cell1", "adnc"] == "High"
    assert "cognition" not in basic

    disease.write_text(f"{header},Cognitive Status\ndonor1,High,No dementia\n")
    extended = data.seaad_annotations(include_cognition=True)
    assert extended.loc["cell1", "cognition"] == "No dementia"
    assert extended.drop(columns="cognition").equals(basic)
