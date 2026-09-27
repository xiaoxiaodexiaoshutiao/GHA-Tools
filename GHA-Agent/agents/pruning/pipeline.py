import time
from typing import Dict, List, Set, Tuple, Any
from .types import LogLine, AnchorResult, ProtectResult, PruningResult
from .preprocessor import Preprocessor
from .anchor_finder import AnchorFinder
from .protector import Protector
from .safe_pruner import SafePruner
from .prune_proposer import PruneProposer, StrictPruneProposer
from .prune_critic import PruneCritic
from .emitter import Emitter, subtract_ranges


class PruningPipeline:
    def __init__(self,
                 llm_client=None,
                 config: Dict[str, Any] = None,
                 enable_llm_steps: bool = True,
                 verbose: bool = True,
                 strict_mode: bool = False):
        self.llm_client = llm_client
        self.config = config or {}
        self.enable_llm_steps = enable_llm_steps
        self.verbose = verbose
        self.strict_mode = strict_mode

        anchor_config = self.config.get('anchor_finder', {})
        protector_config = self.config.get('protector', {})
        safe_pruner_config = self.config.get('safe_pruner', {})

        self.preprocessor = Preprocessor()

        self.anchor_finder = AnchorFinder(
            min_anchors_threshold=anchor_config.get('min_anchors_threshold', 3),
            llm_client=llm_client if enable_llm_steps else None
        )

        self.protector = Protector(
            k_before=protector_config.get('k_before', 4),
            k_after=protector_config.get('k_after', 6),
            soft_window=protector_config.get('soft_window', 10)
        )

        self.safe_pruner = SafePruner(
            min_distance_from_anchor=safe_pruner_config.get('min_distance_from_anchor', 20),
            max_consecutive_duplicates=safe_pruner_config.get('max_consecutive_duplicates', 3)
        )

        if strict_mode:
            self.prune_proposer = StrictPruneProposer(
                llm_client=llm_client if enable_llm_steps else None
            )
            if verbose:
                print("[PruningPipeline] Using strict pruning mode (only delete completely irrelevant lines)")
        else:
            self.prune_proposer = PruneProposer(
                llm_client=llm_client if enable_llm_steps else None
            )
            if verbose:
                print("[PruningPipeline] Using loose pruning mode (original method)")

        self.prune_critic = PruneCritic(
            llm_client=llm_client if enable_llm_steps else None,
            validation_mode=self.config.get('validation_mode', 'evidence')
        )

        self.emitter = Emitter()

    def process(self, log_content: str,
               block_start: int,
               block_end: int,
               block_id: str = "0",
               log_metadata: Dict[str, str] = None) -> Dict[str, Any]:
        start_time = time.time()

        result = {
            'status': 'success',
            'irrelevant_ranges': [],
            'delete_lines': [],
            'kept_ranges': [(block_start, block_end)],
            'statistics': {},
            'timing': {}
        }

        try:
            t0_start = time.time()
            self._log("T0: Preprocessing...")

            self.preprocessor.block_id = block_id
            log_lines = self.preprocessor.process(log_content, block_start)

            result['timing']['t0_preprocess'] = time.time() - t0_start
            self._log(f"  Preprocessing completed: {len(log_lines)} lines")

            line_dict = {line.no: line for line in log_lines}
            all_line_nos = set(line_dict.keys())

            t1_start = time.time()
            self._log("T1: Finding anchors...")

            anchors = self.anchor_finder.find(log_lines)

            result['timing']['t1_anchor_find'] = time.time() - t1_start
            self._log(f"  Found {len(anchors.anchors_hard)} hard anchors, {len(anchors.anchors_soft)} soft anchors")

            if anchors.is_empty():
                self._log("  No anchors found, keeping entire block")
                return result

            t2_start = time.time()
            self._log("T2: Building protection zones...")

            protect = self.protector.build(log_lines, anchors)

            result['timing']['t2_protect'] = time.time() - t2_start
            self._log(f"  Hard protect: {len(protect.hard_protect_set)} lines, "
                     f"Soft protect: {len(protect.soft_protect_set)} lines, "
                     f"Candidates: {len(protect.candidate_set)} lines")

            t3_start = time.time()
            self._log("T3: Safe pre-pruning...")

            predelete_ids = self.safe_pruner.identify(log_lines, protect)

            result['timing']['t3_safe_prune'] = time.time() - t3_start
            self._log(f"  Pre-pruned: {len(predelete_ids)} lines")

            llm_delete_ids = set()
            if self.enable_llm_steps and self.llm_client:
                t4_start = time.time()
                self._log("T4: LLM deletion proposal...")

                llm_delete_ids = self.prune_proposer.propose(log_lines, anchors, protect)

                result['timing']['t4_llm_propose'] = time.time() - t4_start
                self._log(f"  LLM proposed deletion: {len(llm_delete_ids)} lines")
            else:
                self._log("T4: Skipping LLM deletion proposal (not enabled)")

            restore_ids = set()
            all_delete_ids = predelete_ids | llm_delete_ids

            if self.enable_llm_steps and all_delete_ids:
                t5_start = time.time()
                self._log("T5: Validation recovery...")

                restore_ids = self.prune_critic.validate(log_lines, all_delete_ids, protect)

                result['timing']['t5_validate'] = time.time() - t5_start
                self._log(f"  Need to restore: {len(restore_ids)} lines")
            else:
                self._log("T5: Skipping validation recovery")

            t6_start = time.time()
            self._log("T6: Merging output...")

            pruning_result = self.emitter.merge(
                predelete_ids=predelete_ids,
                llm_delete_ids=llm_delete_ids,
                restore_ids=restore_ids,
                hard_protect_set=protect.hard_protect_set,
                all_line_nos=all_line_nos
            )

            result['timing']['t6_merge'] = time.time() - t6_start

            irrelevant_ranges = self.emitter.format_as_irrelevant_ranges(
                pruning_result.delete_lines
            )

            kept_ranges = subtract_ranges(
                (block_start, block_end),
                irrelevant_ranges
            )

            result['irrelevant_ranges'] = irrelevant_ranges
            result['delete_lines'] = pruning_result.delete_lines
            result['kept_ranges'] = kept_ranges
            result['statistics'] = pruning_result.statistics
            result['statistics']['anchors_hard'] = len(anchors.anchors_hard)
            result['statistics']['anchors_soft'] = len(anchors.anchors_soft)

            total_time = time.time() - start_time
            result['timing']['total'] = total_time

            self._log(f"  Final deletion: {len(pruning_result.delete_lines)} lines, "
                     f"Kept: {len(kept_ranges)} ranges")
            self._log(f"Pruning completed, time taken: {total_time:.2f}s")

        except Exception as e:
            self._log(f"Pruning failed: {e}", level="error")
            result['status'] = 'failed'
            result['error'] = str(e)
            import traceback
            traceback.print_exc()

        return result

    def process_from_lines(self, log_lines: List[str],
                          block_start: int,
                          block_end: int,
                          block_id: str = "0",
                          log_metadata: Dict[str, str] = None) -> Dict[str, Any]:
        block_content = "\n".join(log_lines[block_start-1:block_end])

        return self.process(
            log_content=block_content,
            block_start=block_start,
            block_end=block_end,
            block_id=block_id,
            log_metadata=log_metadata
        )

    def _log(self, message: str, level: str = "info"):
        if self.verbose:
            prefix = "[PruningPipeline]"
            if level == "error":
                print(f"❌ {prefix} {message}")
            elif level == "warning":
                print(f"⚠️  {prefix} {message}")
            else:
                print(f"ℹ️  {prefix} {message}")


class PruningPipelineFactory:
    @staticmethod
    def create(enable_llm: bool = True,
               verbose: bool = True,
               strict_mode: bool = False) -> PruningPipeline:
        llm_client = None
        config = {}

        if enable_llm:
            try:
                from core.config_loader import config_loader

                try:
                    pipeline_config = config_loader.get_agent_config('pruning_pipeline')
                    config = pipeline_config

                    proposer_config = pipeline_config.get('prune_proposer', {})
                    if 'model' in proposer_config:
                        from core.llm_client import LLMClient
                        llm_client = LLMClient(proposer_config)
                except Exception:
                    extraction_config = config_loader.get_agent_config('extraction_agents')
                    agent_config = extraction_config.get('pruning_agent_1', {})
                    if 'model' in agent_config:
                        from core.llm_client import LLMClient
                        llm_client = LLMClient(agent_config)

            except Exception as e:
                print(f"Warning: Failed to load LLM client: {e}")

        return PruningPipeline(
            llm_client=llm_client,
            config=config,
            enable_llm_steps=enable_llm and llm_client is not None,
            verbose=verbose,
            strict_mode=strict_mode
        )

    @staticmethod
    def create_simple(verbose: bool = True) -> PruningPipeline:
        return PruningPipeline(
            llm_client=None,
            config={},
            enable_llm_steps=False,
            verbose=verbose
        )
