"""read_data.py의 graph_mode(tri/bipartite) 경로 분기 단위 테스트 (Issue #34).

read_all_data_tri()는 파일 I/O가 섞여 있어 통째로 테스트하기 어렵다 —
"어떤 graph_mode면 어떤 경로/persona_num을 쓰는가"라는 순수 로직만
_resolve_graph_paths()로 분리해 테스트한다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "baselines" / "lgcn3"))

from read_data import _resolve_graph_paths  # noqa: E402


class TestResolveGraphPaths:
    def test_tri_mode_uses_tri_files_and_mba_persona_num(self):
        path_t2p, path_u2p, persona_num = _resolve_graph_paths(
            dataset="MBA", graph_mode="tri", approximate=False, dir_="DIR/"
        )
        assert path_t2p == "DIR/tri_graph_tidx2pidx.json"
        assert path_u2p == "DIR/tri_graph_uidx2pidx.json"
        assert persona_num == 6

    def test_tri_mode_approximate_uses_approximate_u2p_file(self):
        _, path_u2p, _ = _resolve_graph_paths(
            dataset="MBA", graph_mode="tri", approximate=True, dir_="DIR/"
        )
        assert path_u2p == "DIR/tri_graph_uidx2pidx_app_e_0.1.json"

    def test_bipartite_mode_uses_bipartite_files_and_zero_personas(self):
        path_t2p, path_u2p, persona_num = _resolve_graph_paths(
            dataset="MBA", graph_mode="bipartite", approximate=False, dir_="DIR/"
        )
        assert path_t2p == "DIR/bipartite_graph_tidx2pidx.json"
        assert path_u2p == "DIR/bipartite_graph_uidx2pidx.json"
        assert persona_num == 0

    def test_unknown_graph_mode_raises(self):
        import pytest

        with pytest.raises(ValueError):
            _resolve_graph_paths(
                dataset="MBA", graph_mode="not_a_mode", approximate=False, dir_="DIR/"
            )
