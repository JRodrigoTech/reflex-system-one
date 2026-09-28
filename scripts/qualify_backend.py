"""Run a real backend qualification inside its repository-owned environment."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["REFLEX_ROOT"] = str(ROOT)


def load_laya_qualification_service(device: str = "cuda", variant: str | None = None):
    """Load the statically selected Laya checkpoint, independent of user config."""
    from reflex.backends.laya import LayaBackend
    from reflex.model_manager import ModelManager
    from reflex.registry import load_registry
    from reflex.service import DecisionService

    registry = load_registry(ROOT)
    spec = registry["laya"]
    variant = variant or spec.default_variant
    if variant not in spec.variants:
        raise ValueError("Laya qualification variant must be selected from the static registry.")
    manager = ModelManager(ROOT, registry)
    if not manager.verify(spec.model_id, variant):
        raise RuntimeError("The exact pinned Laya checkpoint is not fully verified.")
    backend = LayaBackend.load(spec, manager.destination(spec.model_id, variant), variant, device=device)
    return DecisionService(registry, spec.model_id, backend, variant)


def qualify_laya(device: str = "cuda", variant: str = "multilingual") -> None:
    from reflex.errors import ReflexError
    from reflex.qualification import write_qualification_record

    # A static --qualify=laya selection must qualify Laya even when normal
    # inference is currently configured for another active model.
    service = load_laya_qualification_service(device, variant)
    fixtures = []
    try:
        service.prepare()
        fixtures.append({"id": "startup-readiness-preparation", "passed": True})
        if variant == "multilingual":
            goldens = [
                ("english-account-access", "The user cannot sign in after resetting their password.",
                 {"type": "choice", "instructions": "Which team should handle this issue?", "criteria": {
                     "account": "The user has trouble logging in or managing their account.",
                     "billing": "The user has a payment or invoice problem."}}, "account"),
                ("spanish-duplicate-charge", "El cliente ha sido cobrado dos veces por la misma factura.",
                 {"type": "choice", "instructions": "¿Qué equipo debe gestionar este problema?", "criteria": {
                     "billing": "El cliente tiene un problema con pagos o facturas.",
                     "technical": "El servicio tiene un fallo técnico."}}, "billing"),
                ("japanese-account-access", "パスワードをリセットした後、アカウントにログインできません。",
                 {"type": "choice", "instructions": "Which team should handle this issue?", "criteria": {
                     "account": "The user has trouble logging in or managing their account.",
                     "billing": "The user has a payment or invoice problem."}}, "account"),
            ]
        elif variant == "typed-decisions":
            # This is an operational/runtime qualification only. Output
            # structure and normalization are checked below; without a
            # reserved validation set, statistical calibration is not claimed.
            goldens = []
        else:
            goldens = [
                ("english-account-access", "The user cannot sign in after resetting their password.",
                 {"type": "choice", "instructions": "Which team should handle this issue?", "criteria": {
                     "account": "The user has trouble logging in or managing their account.",
                     "billing": "The user has a payment or invoice problem."}}, "account"),
                ("english-duplicate-charge", "The customer was charged twice for the same invoice and requests a refund.",
                 {"type": "choice", "instructions": "Which team should handle this issue?", "criteria": {
                     "billing": "Payments, duplicate charges, invoices, and refunds.",
                     "technical": "Product defects and service outages."}}, "billing"),
            ]
        for fixture_id, state, question, expected in goldens:
            result = service.predict({"state": state, "questions": {"route": question}})
            answer = result["answers"]["route"]
            probabilities = list(answer["probabilities"].values())
            passed = (
                answer["choice"] == expected
                and all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in probabilities)
                and math.isclose(math.fsum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-6)
            )
            if not passed:
                raise RuntimeError(f"Laya {variant} {device} golden failed: {fixture_id}")
            fixtures.append({"id": fixture_id, "expected": expected, "actual": answer["choice"], "passed": True})

        if variant != "typed-decisions":
            noul = service.predict({"state": "Please refund the duplicate charge on my last invoice.", "questions": {
                "refund": {"type": "noul", "instructions": "Did the customer request a refund?",
                           "criteria": {"true": "The customer wants the charge returned.",
                                        "false": "The customer did not ask for money back."}}
            }})["answers"]["refund"]
            if noul["noul"] < 0.5:
                raise RuntimeError(f"Laya {variant} {device} Noul golden failed: refund request was missed.")
            fixtures.append({"id": "english-refund-request-noul", "expected": True,
                             "probability": noul["noul"], "passed": True})

        if variant == "english":
            # The pinned upstream checkpoint marks this bucket uncalibrated:
            # its 11+ temperature is outside the supported range and upstream
            # clamps it to 0.5. Verify the decision still returns a valid
            # ordered distribution without representing it as calibrated.
            wide_criteria = {f"option-{index:02d}": f"Choice description {index}." for index in range(1, 12)}
            wide_answer = service.predict({"state": "Choose the option that best describes a routine request.",
                                           "questions": {"wide": {"type": "choice", "instructions": "Choose one.",
                                                                    "criteria": wide_criteria}}})["answers"]["wide"]
            wide_probabilities = wide_answer["probabilities"]
            values = list(wide_probabilities.values())
            if (list(wide_probabilities) != list(wide_criteria)
                    or wide_answer["choice"] not in wide_criteria
                    or not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values)
                    or not math.isclose(math.fsum(values), 1.0, rel_tol=0.0, abs_tol=1e-6)):
                raise RuntimeError("Laya english 11+ Choice fallback returned an invalid distribution")
            fixtures.append({"id": "eleven-option-choice-runtime-fallback", "passed": True,
                             "probability_calibration": "upstream marks choice:11+ confidence uncalibrated"})

        mixed_locale = "es" if variant == "multilingual" else "en"
        mixed_message = ("No puedo iniciar sesión desde que cambié mi contraseña."
                         if variant == "multilingual" else
                         "I cannot sign in after changing my password.")
        mixed = service.predict({"state": {"locale": mixed_locale, "conversation": [
            {"role": "user", "content": mixed_message}
        ]}, "questions": {
            "account": {"type": "noul", "instructions": {"goal": "Decide whether this is an account access problem."},
                        "criteria": {"true": {"label": "account access"}, "false": ["other issue"]}},
            "team": {"type": "choice", "instructions": ["Choose the responsible team."], "criteria": {
                "account": {"description": "Login and account settings"}, "billing": "Payments and invoices",
                "technical": ["Service outage", "broken feature"]}},
            "urgency": {"type": "score", "instructions": "Rate the urgency from the evidence.",
                        "criteria": ["low", "medium", {"when": "customer blocked", "level": "high"}]},
            "fallback": {"type": "noul"},
        }})
        if list(mixed["answers"]) != ["account", "team", "urgency", "fallback"]:
            raise RuntimeError("Laya mixed question IDs were not preserved")
        if variant == "typed-decisions":
            for question_id in ("account", "fallback"):
                value = mixed["answers"][question_id].get("noul")
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise RuntimeError(f"Laya typed-decisions returned an invalid Noul probability for {question_id}")
        score = mixed["answers"]["urgency"]
        if score["legend"] != {"0": "low", "1": "medium", "2": '{"level":"high","when":"customer blocked"}'}:
            raise RuntimeError("Laya structured Score legend normalization changed")
        choice = mixed["answers"]["team"]
        if (choice["choice"] not in {"account", "billing", "technical"}
                or list(choice["probabilities"]) != ["account", "billing", "technical"]):
            raise RuntimeError("Laya mixed Choice output changed its option set or ordering")
        expected_score = math.fsum(int(key) * value for key, value in score["probabilities"].items())
        if not math.isclose(score["score"], expected_score, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("Laya Score result is not zero-based probability weighted")
        for question_id, answer in mixed["answers"].items():
            if answer["type"] != "noul":
                probabilities = list(answer["probabilities"].values())
                if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in probabilities):
                    raise RuntimeError(f"Laya emitted invalid probabilities for {question_id}")
                if not math.isclose(math.fsum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-6):
                    raise RuntimeError(f"Laya probabilities did not normalize for {question_id}")
        fixtures.append({"id": "mixed-structured-primitives", "passed": True,
                         "answers": ["noul", "choice", "score"], "structured_inputs": True,
                         **({"probability_calibration": "not evaluated; no validation dataset supplied"}
                            if variant == "typed-decisions" else {})})

        try:
            service.predict({"state": "context " * 1800, "questions": {"q": {"type": "noul"}}})
        except ReflexError as exc:
            if exc.code != "request_exceeds_backend_limits":
                raise RuntimeError("Laya oversized request produced the wrong owned error") from None
        else:
            raise RuntimeError("Laya accepted a request beyond its configured context limit")
        fixtures.append({"id": "configured-context-rejection", "passed": True})

        write_qualification_record(
            "laya", root=ROOT, variant=variant, backend=service.backend, fixtures=fixtures
        )
    finally:
        service.backend.close()


def qualify_decider(model_id: str, device: str = "cuda") -> None:
    """Qualify one pinned Decider checkpoint using its bundled eager adapter."""
    import torch

    from reflex.backends.decider import DeciderBackend
    from reflex.errors import ReflexError
    from reflex.model_manager import ModelManager
    from reflex.qualification import write_qualification_record
    from reflex.registry import load_registry
    from reflex.service import DecisionService

    registry = load_registry(ROOT)
    spec = registry[model_id]
    manager = ModelManager(ROOT, registry)
    if not manager.verify(model_id):
        raise RuntimeError(f"The exact pinned {model_id} checkpoint is not fully verified.")
    backend = DeciderBackend.load(spec, manager.destination(model_id), device=device)
    service = DecisionService(registry, model_id, backend)
    fixtures: list[dict[str, object]] = []
    try:
        cases = [
            {
                "id": "duplicate-invoice-routes-to-billing",
                "request": {
                    "state": "The customer paid one invoice once, but the same card was charged twice for it.",
                    "questions": {"route": {"type": "choice", "instructions": "Which team should resolve this issue?", "criteria": {
                        "billing": "Payments, duplicate charges, and invoices.",
                        "account": "Login and account profile problems.",
                        "technical": "Product bugs and service outages.",
                    }}},
                },
                "expected": "billing",
            },
            {
                "id": "duplicate-charge-noul",
                "request": {
                    "state": "The same purchase appears twice on the customer's card statement.",
                    "questions": {"duplicate": {"type": "noul", "instructions": "Was the customer charged more than once for the same purchase?"}},
                },
                "expected": True,
            },
            {
                "id": "blocked-production-score",
                "request": {
                    "state": "A production outage has locked every user out of the service and the customer cannot process any payments.",
                    "questions": {"severity": {"type": "score", "instructions": "Rate the severity based on the impact described.", "criteria": [
                        "No meaningful impact.", "A minor inconvenience with a workaround.",
                        "A serious issue affecting some users.", "A critical outage preventing all customers from using the service.",
                    ]}},
                },
                "expected_min_score": 2.0,
            },
        ]
        for case in cases:
            request = case["request"]
            result = service.predict(request)
            if list(result["answers"]) != list(request["questions"]):
                raise RuntimeError(f"Decider did not preserve question order for {case['id']}.")
            answer = result["answers"][next(iter(request["questions"]))]
            if answer["type"] == "choice":
                passed = answer["choice"] == case["expected"]
                probabilities = list(answer["probabilities"].values())
            elif answer["type"] == "noul":
                passed = (answer["noul"] >= 0.5) is case["expected"]
                probabilities = [1.0 - answer["noul"], answer["noul"]]
            else:
                passed = answer["score"] >= case["expected_min_score"]
                probabilities = list(answer["probabilities"].values())
            passed = passed and all(math.isfinite(float(p)) and 0 <= p <= 1 for p in probabilities)
            passed = passed and math.isclose(math.fsum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-6)
            if not passed:
                raise RuntimeError(f"Pinned {model_id} golden failed: {case['id']}.")
            fixtures.append({"id": case["id"], "expected": case.get("expected", case.get("expected_min_score")),
                             "actual": answer.get("choice", answer.get("noul", answer.get("score"))), "passed": True})

        # Exercise structured state/instructions/criteria, mixed answer types,
        # stable ID ordering, and the pinned per-type temperature map together.
        mixed = service.predict({"state": {"account": {"status": "locked out"}, "events": ["reset link expired"]}, "questions": {
            "access": {"type": "noul", "instructions": {"task": "Decide if account access is blocked."}, "criteria": {
                "true": {"meaning": "The user cannot sign in."}, "false": ["Account access works"]}},
            "route": {"type": "choice", "instructions": ["Select the team."], "criteria": {
                "account": {"meaning": "Login and profile help."}, "billing": "Payments and invoices."}},
            "severity": {"type": "score", "instructions": "Rate customer impact.", "criteria": ["low", "medium", {"level": "high", "when": "blocked"}]},
        }})
        if list(mixed["answers"]) != ["access", "route", "severity"]:
            raise RuntimeError("Decider mixed request changed question IDs or their order.")
        score = mixed["answers"]["severity"]
        if score["legend"] != {"0": "low", "1": "medium", "2": '{"level":"high","when":"blocked"}'}:
            raise RuntimeError("Decider structured Score descriptions were not normalized deterministically.")
        fixtures.append({"id": "mixed-structured-primitives", "passed": True,
                         "answers": [answer["type"] for answer in mixed["answers"].values()],
                         "temperature_by_type": {"choice": backend.engine.T_by_type.get("choice"),
                                                  "noul": backend.engine.T_by_type.get("noul"),
                                                  "score": backend.engine.T_by_type.get("score")}})

        # The backend must reject overflow before inference, even though the
        # upstream renderer has truncation behavior at its configured cap.
        too_long = {"state": "overflow " * 40000, "questions": {"q": {"type": "noul"}}}
        try:
            service.predict(too_long)
        except ReflexError as exc:
            if exc.code != "context_too_long":
                raise RuntimeError("Decider context overflow returned the wrong owned error.") from None
        else:
            raise RuntimeError("Decider accepted input beyond the pinned complete-question token limit.")
        fixtures.append({"id": "context-limit-rejection", "passed": True, "limit": backend.context_limit})

        write_qualification_record(model_id, root=ROOT, backend=backend, fixtures=fixtures)
    finally:
        backend.close()


def qualify_semantic_router(model_id: str) -> None:
    """Qualify one exact Sol/Nox checkpoint through native Windows CUDA."""
    from reflex.backends.semantic_router import SemanticRouterBackend
    from reflex.errors import ReflexError
    from reflex.model_manager import ModelManager
    from reflex.qualification import write_qualification_record
    from reflex.registry import load_registry
    from reflex.service import DecisionService

    registry = load_registry(ROOT)
    spec = registry[model_id]
    manager = ModelManager(ROOT, registry)
    if not manager.verify(model_id):
        raise RuntimeError(f"The exact pinned {model_id} checkpoint is not fully verified.")
    backend = SemanticRouterBackend.load(
        spec,
        manager.destination(model_id),
        ROOT / "manifests" / "qualifications" / f"{model_id}.json",
        device="cuda:0",
        allow_unqualified=True,
    )
    service = DecisionService(registry, model_id, backend)
    fixtures: list[dict[str, object]] = []
    try:
        # This invoice/refund example is shared by the upstream Sol and Nox
        # model cards. It qualifies the typed head rather than text generation.
        state = "Customer reports a duplicate charge and asks for a refund."
        cases = [
            {
                "id": "duplicate-invoice-choice",
                "request": {
                    "state": state,
                    "questions": {"route": {"type": "choice", "instructions": "Which team should handle this issue?", "criteria": {
                        "billing": "Payments, duplicate charges, invoices, and refunds.",
                        "technical": "Product defects and service outages.",
                    }}},
                },
                "expected": "billing",
            },
            {
                "id": "refund-requested-noul",
                "request": {
                    "state": state,
                    "questions": {"refund_requested": {"type": "noul", "instructions": "Did the customer request a refund?"}},
                },
                "expected": True,
            },
            {
                "id": "duplicate-invoice-score",
                "request": {
                    "state": state,
                    "questions": {"urgency": {"type": "score", "instructions": "Rate the urgency of this issue.", "criteria": [
                        "Routine account question.",
                        "Payment issue that needs customer support.",
                        "Critical outage affecting all customers.",
                    ]}},
                },
                "score_range": (0.25, 1.75),
            },
        ]
        for case in cases:
            result = service.predict(case["request"])
            question_id = next(iter(case["request"]["questions"]))
            answer = result["answers"][question_id]
            if answer["type"] == "choice":
                actual = answer["choice"]
                probabilities = list(answer["probabilities"].values())
                passed = actual == case["expected"]
            elif answer["type"] == "noul":
                actual = answer["noul"] >= 0.5
                probabilities = [1.0 - answer["noul"], answer["noul"]]
                passed = actual is case["expected"]
            else:
                actual = answer["score"]
                probabilities = list(answer["probabilities"].values())
                lower, upper = case["score_range"]
                passed = lower <= actual <= upper
            passed = passed and all(math.isfinite(float(value)) and 0.0 <= value <= 1.0 for value in probabilities)
            passed = passed and math.isclose(math.fsum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-6)
            if not passed:
                raise RuntimeError(f"Pinned {model_id} NVIDIA golden failed: {case['id']}.")
            fixtures.append({"id": case["id"], "expected": case.get("expected", case.get("score_range")),
                             "actual": actual, "passed": True})

        mixed_request = {
            "state": {"summary": state, "product_status": ["healthy", "no outage"]},
            "questions": {
                "duplicate": {"type": "noul", "instructions": {"task": "Decide whether the same invoice was charged twice."}, "criteria": {
                    "true": {"meaning": "The invoice was charged more than once."},
                    "false": ["The invoice was charged once or the evidence is insufficient."],
                }},
                "team": {"type": "choice", "instructions": ["Choose the responsible team."], "criteria": {
                    "billing": {"meaning": "Payments, invoices, and refunds."},
                    "technical": ["Product bugs", "service outages"],
                }},
                "urgency": {"type": "score", "instructions": "Rate the urgency using the ordered scale.", "criteria": [
                    "Routine request.", {"level": "payment support", "rank": 1}, "Critical outage."
                ]},
            },
        }
        mixed = service.predict(mixed_request)
        if list(mixed["answers"]) != ["duplicate", "team", "urgency"]:
            raise RuntimeError("CandidateHead changed mixed question IDs or order.")
        score = mixed["answers"]["urgency"]
        if score["legend"] != {"0": "Routine request.", "1": '{"level":"payment support","rank":1}', "2": "Critical outage."}:
            raise RuntimeError("CandidateHead structured Score legend normalization changed.")
        for question_id, answer in mixed["answers"].items():
            probabilities = (
                list(answer["probabilities"].values())
                if answer["type"] != "noul"
                else [1.0 - answer["noul"], answer["noul"]]
            )
            if (not all(math.isfinite(float(value)) and 0.0 <= value <= 1.0 for value in probabilities)
                    or not math.isclose(math.fsum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-6)):
                raise RuntimeError(f"CandidateHead emitted invalid probabilities for {question_id}.")
        expected_score = math.fsum(int(key) * value for key, value in score["probabilities"].items())
        if not math.isclose(score["score"], expected_score, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("CandidateHead Score is not zero-based probability weighted.")
        fixtures.append({"id": "mixed-structured-primitives", "passed": True,
                         "answers": [answer["type"] for answer in mixed["answers"].values()]})

        try:
            service.predict({"state": "overflow " * 18000, "questions": {"q": {"type": "noul"}}})
        except ReflexError as exc:
            if exc.code != "context_too_long":
                raise RuntimeError("CandidateHead context overflow returned the wrong owned error.") from None
        else:
            raise RuntimeError("CandidateHead accepted input beyond its pinned complete-question token limit.")
        fixtures.append({"id": "context-limit-rejection", "passed": True, "limit": 16384})

        write_qualification_record(model_id, root=ROOT, backend=backend, fixtures=fixtures)
    finally:
        backend.close()


def qualify_lux() -> None:
    """Attempt Lux qualification only on native Windows and matching hardware."""
    import torch

    from reflex.backends.lux import LuxBackend, validate_lux_windows_cuda
    from reflex.errors import ReflexError
    from reflex.model_manager import ModelManager
    from reflex.qualification import write_qualification_record
    from reflex.registry import load_registry
    from reflex.service import DecisionService

    if os.name != "nt":
        raise RuntimeError("Lux qualification requires the native Windows runtime.")
    registry = load_registry(ROOT)
    spec = registry["lux-9b"]
    manager = ModelManager(ROOT, registry)
    if not manager.verify("lux-9b"):
        raise RuntimeError("The exact pinned Lux checkpoint is not fully verified.")
    model_dir = manager.destination("lux-9b")
    runtime = json.loads((model_dir / "runtime.json").read_text(encoding="utf-8"))
    profile = runtime.get("normalization_profile")
    if not isinstance(profile, dict):
        raise RuntimeError("The pinned Lux checkpoint has no supported runtime profile.")
    if not torch.cuda.is_available():
        raise RuntimeError("Lux qualification requires a native Windows NVIDIA CUDA device.")
    device = "cuda:0"
    properties = torch.cuda.get_device_properties(device)
    try:
        free_vram_bytes, _ = torch.cuda.mem_get_info(device)
    except Exception:
        raise RuntimeError("Available NVIDIA GPU memory could not be checked safely.") from None
    required_weight_bytes = spec.estimated_bytes.get("default", 0)
    try:
        validate_lux_windows_cuda(
            profile,
            properties,
            os_name=os.name,
            cuda_available=bool(torch.cuda.is_available()),
            cuda_version=getattr(torch.version, "cuda", None),
            hip_version=getattr(torch.version, "hip", None),
            bf16_supported=bool(torch.cuda.is_bf16_supported()),
            free_vram_bytes=free_vram_bytes,
            required_vram_bytes=required_weight_bytes,
        )
    except ReflexError as exc:
        raise RuntimeError(exc.message) from None

    backend = LuxBackend.load(
        spec,
        model_dir,
        ROOT / "manifests" / "qualifications" / "lux-9b.json",
        device=device,
        allow_unqualified=True,
        required_vram_bytes=required_weight_bytes,
    )
    service = DecisionService(registry, "lux-9b", backend)
    fixtures: list[dict[str, object]] = []
    try:
        cases = [
            {
                "id": "duplicate-invoice-choice",
                "request": {"state": "The customer was charged twice for one invoice and needs a refund.", "questions": {
                    "route": {"type": "choice", "instructions": "Which team should handle this issue?", "criteria": {
                        "billing": "Payments, duplicate charges, and refunds.",
                        "technical": "Product faults and service outages.",
                    }}
                }},
                "expected": "billing",
            },
            {
                "id": "refund-requested-noul",
                "request": {"state": "The customer asks for the duplicate payment to be returned.", "questions": {
                    "refund": {"type": "noul", "instructions": "Did the customer request a refund?"}
                }},
                "expected": True,
            },
            {
                "id": "payment-support-score",
                "request": {"state": "The customer was charged twice and needs support.", "questions": {
                    "urgency": {"type": "score", "instructions": "Rate the urgency.", "criteria": [
                        "Routine question.", "Payment issue requiring support.", "Critical outage affecting all customers."
                    ]}
                }},
                "score_range": (0.25, 1.75),
            },
        ]
        for case in cases:
            result = service.predict(case["request"])
            question_id = next(iter(case["request"]["questions"]))
            answer = result["answers"][question_id]
            if answer["type"] == "choice":
                actual = answer["choice"]
                probabilities = list(answer["probabilities"].values())
                passed = actual == case["expected"]
            elif answer["type"] == "noul":
                actual = answer["noul"] >= 0.5
                probabilities = [1.0 - answer["noul"], answer["noul"]]
                passed = actual is case["expected"]
            else:
                actual = answer["score"]
                probabilities = list(answer["probabilities"].values())
                low, high = case["score_range"]
                passed = low <= actual <= high
            passed = passed and all(math.isfinite(float(value)) and 0.0 <= value <= 1.0 for value in probabilities)
            passed = passed and math.isclose(math.fsum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-6)
            if not passed:
                raise RuntimeError(f"Pinned Lux native Windows golden failed: {case['id']}.")
            fixtures.append({"id": case["id"], "actual": actual, "passed": True})

        mixed_request = {"state": {"summary": "A duplicate invoice charge is reported.", "contact": ["refund requested"]},
                         "questions": {
            "duplicate": {"type": "noul", "instructions": {"task": "Was the same invoice charged twice?"}},
            "route": {"type": "choice", "instructions": ["Choose the team."], "criteria": {
                "billing": {"meaning": "Payments and invoices."},
                "technical": ["Product faults", "service outages"],
            }},
            "urgency": {"type": "score", "instructions": "Rate the customer impact.", "criteria": [
                "Routine request.", {"level": "payment support"}, "Critical outage."
            ]},
        }}
        mixed = service.predict(mixed_request)
        if list(mixed["answers"]) != ["duplicate", "route", "urgency"]:
            raise RuntimeError("Lux changed mixed question IDs or their order.")
        for question_id, answer in mixed["answers"].items():
            probabilities = (list(answer["probabilities"].values()) if answer["type"] != "noul"
                             else [1.0 - answer["noul"], answer["noul"]])
            if (not all(math.isfinite(float(value)) and 0.0 <= value <= 1.0 for value in probabilities)
                    or not math.isclose(math.fsum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-6)):
                raise RuntimeError(f"Lux emitted invalid probabilities for {question_id}.")
        score = mixed["answers"]["urgency"]
        expected_score = math.fsum(int(key) * value for key, value in score["probabilities"].items())
        if not math.isclose(score["score"], expected_score, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("Lux Score is not zero-based probability weighted.")
        fixtures.append({"id": "mixed-structured-primitives", "passed": True,
                         "answers": [answer["type"] for answer in mixed["answers"].values()]})

        try:
            service.predict({"state": "overflow " * 18000, "questions": {"q": {"type": "noul"}}})
        except ReflexError as exc:
            if exc.code != "context_too_long":
                raise RuntimeError("Lux context overflow returned the wrong owned error.") from None
        else:
            raise RuntimeError("Lux accepted input beyond its 16,384-token limit.")
        fixtures.append({"id": "context-limit-rejection", "passed": True, "limit": 16384})
        write_qualification_record("lux-9b", root=ROOT, backend=backend, fixtures=fixtures)
    finally:
        backend.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("laya", "decider-2b", "decider-4b", "sol-2b", "nox-4b", "lux-9b"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda",
                        help="Select the execution device for Laya or Decider qualification.")
    parser.add_argument("--variant", choices=("multilingual", "english", "typed-decisions"), default=None,
                        help="Select a static Laya checkpoint variant for device-specific runtime qualification.")
    args = parser.parse_args()
    model = args.model
    device = args.device
    if model == "laya":
        variant = args.variant or "multilingual"
        qualify_laya(device, variant)
        if variant == "typed-decisions":
            print(f"Laya {variant} {device.upper()} runtime qualification passed; statistical probability calibration remains unverified.")
        else:
            print(f"Laya {variant} {device.upper()} qualification passed.")
        return 0
    if args.variant is not None:
        raise SystemExit("--variant is supported only when qualifying Laya.")
    if model in {"decider-2b", "decider-4b"}:
        qualify_decider(model, device)
        print(f"{model} {device.upper()} qualification passed.")
        return 0
    if device != "cuda":
        raise SystemExit(f"{model} does not support CPU qualification.")
    if model in {"sol-2b", "nox-4b"}:
        qualify_semantic_router(model)
        print(f"{model} NVIDIA qualification passed.")
        return 0
    qualify_lux()
    print("lux-9b native Windows NVIDIA qualification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
