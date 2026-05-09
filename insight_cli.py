#!/usr/bin/env python3
"""
FM-Oracle1 Insight Engine CLI

CLI interface for the FM-Oracle1 Insight Engine — routes tasks to the optimal
model voice and hardware backend combination.

Usage:
    # Probe available FLUX hardware backends
    python insight_cli.py probe

    # Route a task to the best model + backend
    python insight_cli.py route "rust constraint solver" --signature G-D-I-S-S

    # Route a task with signature auto-detection
    python insight_cli.py route "prove a theorem in Coq" -v

    # Batch route multiple tasks
    python insight_cli.py batch "task1" "task2" --signatures "G-D-I-S-S" "g-d-I-s-s"

    # Show detailed model rankings for a task
    python insight_cli.py rank "deep math proof"

    # Display FLUX bytecode for a task
    python insight_cli.py bytecode "matrix multiplication" --signature G-D-I-S-S
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent dir to path for direct execution
sys.path.insert(0, str(Path(__file__).parent))

from insight_engine import (
    BackendProbe,
    VoiceMatcher,
    InsightRouter,
    encode_task_to_signature,
    generate_flux_bytecode,
    format_bytecode,
    hamming_distance,
    MODEL_VOICE_SIGNATURES,
    FLUX_OPCODES,
)


def cmd_probe(args):
    """Probe all available FLUX hardware backends."""
    probe = BackendProbe(verbose=args.verbose)
    results = probe.probe_all()

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(probe.summary())


def cmd_route(args):
    """Route a task to the best model + backend."""
    router = InsightRouter(
        signatures_dir=args.signatures_dir,
        verbose=args.verbose,
    )

    task_sig = args.signature
    if not task_sig:
        task_sig = encode_task_to_signature(args.task)

    result = router.route(args.task, task_sig=task_sig)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"Task: {result['task']}")
        print(f"Task Signature: {result['task_signature']}")
        print(f"Recommended Model:  {result['model']}")
        print(f"Model Confidence:   {result['model_confidence']:.2%}")
        print(f"Recommended Backend: {result['recommended_backend']}")
        print(f"Available Backends:  {', '.join(result['available_backends'])}")
        print(f"Complexity:         {result['complexity']['level']} "
              f"(score: {result['complexity']['score']})")
        print(f"FLUX Opcodes:       {result['flux_opcode_count']}")
        print()
        print(format_bytecode(result['flux_bytecode']))


def cmd_rank(args):
    """Show detailed model voice rankings for a task."""
    matcher = VoiceMatcher(signatures_dir=args.signatures_dir)

    task_sig = args.signature
    if not task_sig:
        task_sig = encode_task_to_signature(args.task)

    rankings = matcher.match_all(task_sig)

    if args.json:
        print(json.dumps([
            {'model': m, 'confidence': round(c, 4)}
            for m, c in rankings
        ], indent=2))
    else:
        print(f"Task: {args.task}")
        print(f"Signature: {task_sig}")
        print()
        print("Model Voice Rankings:")
        print(f"{'Rank':<6} {'Model':<35} {'Confidence':<12} {'Signature Code':<20}")
        print("-" * 75)
        for i, (model, conf) in enumerate(rankings, 1):
            sig = matcher.signature_info(model)
            sig_code = sig.get('signature', '?') if sig else '?'
            bar = '█' * int(conf * 20) + '░' * (20 - int(conf * 20))
            print(f"{i:<6} {model:<35} {conf:.1%}   {bar}  {sig_code}")
        print()
        print(f"Total models ranked: {len(rankings)}")


def cmd_bytecode(args):
    """Generate and display FLUX bytecode for a task."""
    task_sig = args.signature
    if not task_sig:
        task_sig = encode_task_to_signature(args.task)

    bytecode = generate_flux_bytecode(args.task, task_sig)

    if args.json:
        print(json.dumps({
            'task': args.task,
            'signature': task_sig,
            'bytecode': bytecode,
            'opcode_count': len(bytecode),
        }, indent=2))
    else:
        print(f"Task: {args.task}")
        print(f"Signature: {task_sig}")
        print()
        print(format_bytecode(bytecode))
        print()
        print(f"Total: {len(bytecode)} opcodes")


def cmd_compare(args):
    """Compare two signature codes and show Hamming distance."""
    dist = hamming_distance(args.sig_a, args.sig_b)
    max_dist = max(
        len([c for c in args.sig_a.split('-') if c]),
        len([c for c in args.sig_b.split('-') if c]),
        1
    )
    similarity = 1.0 - (dist / max_dist)

    print(f"Signature A: {args.sig_a}")
    print(f"Signature B: {args.sig_b}")
    print(f"Hamming Distance: {dist}")
    print(f"Similarity: {similarity:.1%}")
    print(f"Match: {'✅' if similarity > 0.8 else '⚠️' if similarity > 0.5 else '❌'}")


def cmd_list(args):
    """List all known model voice signatures."""
    if args.json:
        print(json.dumps(MODEL_VOICE_SIGNATURES, indent=2))
    else:
        print("Known Model Voice Signatures:")
        print(f"{'Model':<35} {'Signature':<20} {'Strength':<25}")
        print("-" * 80)
        for model, info in MODEL_VOICE_SIGNATURES.items():
            strength = info.get('strengths', ['general'])[0]
            print(f"{model:<35} {info['signature']:<20} {strength:<25}")

        print()
        print("FLUX 30-Opcode Subset:")
        for i, op in enumerate(FLUX_OPCODES):
            print(f"  0x{i:02X}  {op}")


def cmd_batch(args):
    """Route multiple tasks in batch."""
    router = InsightRouter(
        signatures_dir=args.signatures_dir,
        verbose=args.verbose,
    )

    if args.signatures:
        signatures = args.signatures
        if len(signatures) != len(args.tasks):
            print("Error: --signatures count must match task count",
                  file=sys.stderr)
            sys.exit(1)
        tasks = list(zip(args.tasks, signatures))
    else:
        tasks = args.tasks

    results = router.route_batch(tasks)

    if args.json:
        print(json.dumps(results, indent=2, default=str))
    else:
        for i, (task, result) in enumerate(zip(tasks, results)):
            if isinstance(task, tuple):
                task_desc = task[0]
            else:
                task_desc = task
            print(f"[{i + 1}] {task_desc}")
            print(f"    Model:    {result['model']} "
                  f"(confidence: {result['model_confidence']:.1%})")
            print(f"    Backend:  {result['recommended_backend']}")
            print(f"    Opcodes:  {result['flux_opcode_count']} "
                  f"({result['complexity']['level']})")
            print()


def main():
    parser = argparse.ArgumentParser(
        description='FM-Oracle1 Insight Engine — Route tasks to optimal model '
                    'and hardware backend',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s probe                          # Check available hardware
  %(prog)s route "constraint solver"      # Route a task (auto-signature)
  %(prog)s route "proof" --sig G-D-I-S-S  # Route with explicit signature
  %(prog)s rank "deep math proof"         # Rank model voices
  %(prog)s bytecode "matrix multiply"     # Show FLUX bytecode
  %(prog)s list                           # List known models
        """,
    )

    sub = parser.add_subparsers(dest='command', required=True)

    # probe
    p_probe = sub.add_parser('probe', help='Probe available FLUX hardware backends')
    p_probe.add_argument('-j', '--json', action='store_true',
                         help='Output as JSON')
    p_probe.add_argument('-v', '--verbose', action='store_true',
                         help='Verbose probing output')
    p_probe.set_defaults(func=cmd_probe)

    # route
    p_route = sub.add_parser('route', help='Route a task to best model + backend')
    p_route.add_argument('task', type=str, help='Task description')
    p_route.add_argument('--signature', '-s', type=str, default=None,
                         help='Task signature code (e.g. G-D-I-S-S). '
                              'Auto-detected if omitted.')
    p_route.add_argument('--signatures-dir', '-d', type=str, default=None,
                         help='Directory of casting-call signature JSON files')
    p_route.add_argument('-j', '--json', action='store_true',
                         help='Output as JSON')
    p_route.add_argument('-v', '--verbose', action='store_true',
                         help='Verbose output')
    p_route.set_defaults(func=cmd_route)

    # rank
    p_rank = sub.add_parser('rank', help='Rank model voices for a task')
    p_rank.add_argument('task', type=str, help='Task description')
    p_rank.add_argument('--signature', '-s', type=str, default=None,
                         help='Task signature code')
    p_rank.add_argument('--signatures-dir', '-d', type=str, default=None,
                         help='Directory of casting-call signature JSON files')
    p_rank.add_argument('-j', '--json', action='store_true',
                         help='Output as JSON')
    p_rank.set_defaults(func=cmd_rank)

    # bytecode
    p_bc = sub.add_parser('bytecode', help='Generate FLUX bytecode for a task')
    p_bc.add_argument('task', type=str, help='Task description')
    p_bc.add_argument('--signature', '-s', type=str, default=None,
                       help='Task signature code')
    p_bc.add_argument('-j', '--json', action='store_true',
                       help='Output as JSON')
    p_bc.set_defaults(func=cmd_bytecode)

    # compare
    p_cmp = sub.add_parser('compare', help='Compare two signature codes')
    p_cmp.add_argument('sig_a', type=str, help='First signature code')
    p_cmp.add_argument('sig_b', type=str, help='Second signature code')
    p_cmp.set_defaults(func=cmd_compare)

    # list
    p_list = sub.add_parser('list', help='List known model voice signatures')
    p_list.add_argument('-j', '--json', action='store_true',
                        help='Output as JSON')
    p_list.set_defaults(func=cmd_list)

    # batch
    p_batch = sub.add_parser('batch', help='Batch route multiple tasks')
    p_batch.add_argument('tasks', type=str, nargs='+',
                         help='Task descriptions')
    p_batch.add_argument('--signatures', '-s', type=str, nargs='*', default=None,
                         help='Signature codes, one per task')
    p_batch.add_argument('--signatures-dir', '-d', type=str, default=None,
                         help='Directory of casting-call signature JSON files')
    p_batch.add_argument('-j', '--json', action='store_true',
                         help='Output as JSON')
    p_batch.add_argument('-v', '--verbose', action='store_true',
                         help='Verbose output')
    p_batch.set_defaults(func=cmd_batch)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
