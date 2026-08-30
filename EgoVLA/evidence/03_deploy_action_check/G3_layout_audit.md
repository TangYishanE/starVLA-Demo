# G3: 48-dim layout audit (RESOLVED — no bug)

Question: does `decode.py`'s flat slicing `[trans(2x3) | mano(2x15) | rot6d(2x6)]`
match the model's actual output layout, given `EgoVLA_ActionHeader.py`'s docstring
claims the decoder emits `[left(3+6+15), right(3+6+15)]` and "the framework
re-slices" (which `EgoVLA.py._run` does NOT do)?

Resolution chain (verified 2026-08-30 against EgoVLA_Release@09645b9):

1. `decode.py` is a faithful port: its slicing matches the ORIGINAL EgoVLA
   inference code `human_plan/ego_bench_eval/utils.py::ik_eval_single_step`
   line-for-line:
   - `pred_3d   = pred[:, :6].reshape(-1, 2, 3)`    (wrist trans 2x3)
   - `pred_hand = pred[:, 6:36].reshape(-1, 2, 15)` (MANO 2x15)
   - `pred_rot  = pred[:, 36:].reshape(-1, 2, 12)` -> rot6d_to_rotmat (2x6)

2. The ORIGINAL decoder `VILA/llava/model/ego_vla_decoder/traj_decoder.py`
   (TrajDecoder -> TransformerSplitActV2, same file the vendored
   `EgoVLA_ActionHeader.py` was copied from) emits
   `torch.cat([out_left, out_right], dim=1).reshape(-1, 2*(3+6+15))`.
   The vendored copy produces the identical tensor layout and loads the
   checkpoint's `traj_decoder` weights strict (98/98 keys, verified in the
   starVLA EgoVLA README and re-verified by the successful load here).

3. Therefore the starVLA framework's `predict_action` output has the SAME flat
   layout the original inference code consumes -> `decode.py` slicing is
   consistent with the model output. The `EgoVLA_ActionHeader.py` docstring
   "(3+6+15)" describes per-hand dim membership, not flat order; the "framework
   re-slices" sentence is stale documentation, not implemented behavior.
   Cosmetic only — no code change needed.

Empirical: framework on a random 384x384 image -> (1,30,48), finite, and the
trans/rot6d slices decode to finite pelvis-frame EE poses (see
ee_ik_partial_chain.log).
