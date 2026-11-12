"""Collect actual encrypted run reports, without copying keys or input data."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots",nargs="+",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    for root in args.roots:
        if not root.is_dir(): raise SystemExit(f"report root does not exist: {root}")
    rows=[]
    for root in args.roots:
        for file in sorted(root.rglob("latest_report.json")):
            report=json.loads(file.read_text(encoding="utf-8"))
            manifest=json.loads((file.parent/"manifest.json").read_text(encoding="utf-8"))
            phases=report["phases"]
            keygen=phases["keygen"]["runtime"]
            execution=phases["evaluate"]["runtime"]
            errors=report.get("validation",{}).get("outputs",{})
            rows.append({
                "model":file.parent.name,"backend":report["backend"],
                "model_sha256":report["model_sha256"],
                "security_profile":keygen.get("security_profile",f"SEAL_tc{keygen.get('security_bits',0)}"),
                "ring_dimension":keygen["ring_dimension"],"depth":keygen["depth"],
                "scalar_operations":manifest["scalar_operations"],
                "logic_comparisons":manifest["logic_comparisons"],
                "evaluation_seconds":execution["seconds"],
                "evaluation_process_seconds":phases["evaluate"]["wall_seconds"],
                "keygen_process_seconds":phases["keygen"]["wall_seconds"],
                "ckks_to_logic_seconds":execution.get("ckks_to_logic_seconds",0),
                "logic_seconds":execution.get("logic_seconds",0),
                "logic_to_ckks_seconds":execution.get("logic_to_ckks_seconds",0),
                "logic_sign_calls":execution.get("logic_bootstrap",0),
                "max_abs_error":max((v["max_abs_error"] for v in errors.values()),default=None),
                "validated":report.get("validation",{}).get("passed",False),
                "ciphertext_input_bytes":report["ciphertext_input_bytes"],
                "ciphertext_output_bytes":report["ciphertext_output_bytes"],
                "evaluation_bundle_bytes":report["evaluation_bundle_bytes"],
                "server_has_no_secret_key":report["server_has_no_secret_key"],
            })
    if not rows: raise SystemExit("no completed reports found")
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"results.json").write_text(json.dumps(rows,indent=2)+"\n",encoding="utf-8")
    with (args.output/"results.csv").open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(f"Collected {len(rows)} completed encrypted runs in {args.output.resolve()}")


if __name__=="__main__": main()
