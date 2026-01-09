"""
Synthetic metric validation test.
Tests F1, IoU, precision, recall against hand-calculated expected values.
Confirms smp.metrics formulas are correct.
"""

import torch
import numpy as np
import segmentation_models_pytorch as smp
from pathlib import Path

def test_perfect_prediction():
    """All predictions correct → F1=1.0, IoU=1.0, Precision=1.0, Recall=1.0"""
    print("\n=== Test 1: Perfect Prediction ===")
    preds = torch.ones(1, 1, 4, 4, dtype=torch.int32)
    masks = torch.ones(1, 1, 4, 4, dtype=torch.int32)
    
    tp, fp, fn, tn = smp.metrics.get_stats(preds, masks.long(), mode='binary')
    iou = smp.metrics.iou_score(tp, fp, fn, tn).mean().item()
    f1 = smp.metrics.f1_score(tp, fp, fn, tn).mean().item()
    precision = smp.metrics.precision(tp, fp, fn, tn).mean().item()
    recall = smp.metrics.recall(tp, fp, fn, tn).mean().item()
    
    print(f"TP={tp.item()}, FP={fp.item()}, FN={fn.item()}, TN={tn.item()}")
    print(f"Expected: F1=1.0, IoU=1.0, Precision=1.0, Recall=1.0")
    print(f"Actual:   F1={f1:.4f}, IoU={iou:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
    assert abs(f1 - 1.0) < 1e-6, f"F1 mismatch: {f1}"
    assert abs(iou - 1.0) < 1e-6, f"IoU mismatch: {iou}"
    print("✓ PASS")


def test_all_false_negatives():
    """Predict all 0, truth is all 1 → F1=0, IoU=0, Precision=undefined, Recall=0"""
    print("\n=== Test 2: All False Negatives ===")
    preds = torch.zeros(1, 1, 4, 4, dtype=torch.int32)
    masks = torch.ones(1, 1, 4, 4, dtype=torch.int32)
    
    tp, fp, fn, tn = smp.metrics.get_stats(preds, masks.long(), mode='binary')
    iou = smp.metrics.iou_score(tp, fp, fn, tn).mean().item()
    f1 = smp.metrics.f1_score(tp, fp, fn, tn).mean().item()
    recall = smp.metrics.recall(tp, fp, fn, tn).mean().item()
    
    print(f"TP={tp.item()}, FP={fp.item()}, FN={fn.item()}, TN={tn.item()}")
    print(f"Expected: F1=0.0, IoU=0.0, Recall=0.0")
    print(f"Actual:   F1={f1:.4f}, IoU={iou:.4f}, Recall={recall:.4f}")
    assert abs(f1 - 0.0) < 1e-6, f"F1 mismatch: {f1}"
    assert abs(iou - 0.0) < 1e-6, f"IoU mismatch: {iou}"
    print("✓ PASS")


def test_all_false_positives():
    """Predict all 1, truth is all 0 → F1=0, IoU=0, Precision=0, Recall=undefined"""
    print("\n=== Test 3: All False Positives ===")
    preds = torch.ones(1, 1, 4, 4, dtype=torch.int32)
    masks = torch.zeros(1, 1, 4, 4, dtype=torch.int32)
    
    tp, fp, fn, tn = smp.metrics.get_stats(preds, masks.long(), mode='binary')
    iou = smp.metrics.iou_score(tp, fp, fn, tn).mean().item()
    f1 = smp.metrics.f1_score(tp, fp, fn, tn).mean().item()
    precision = smp.metrics.precision(tp, fp, fn, tn).mean().item()
    
    print(f"TP={tp.item()}, FP={fp.item()}, FN={fn.item()}, TN={tn.item()}")
    print(f"Expected: F1=0.0, IoU=0.0, Precision=0.0")
    print(f"Actual:   F1={f1:.4f}, IoU={iou:.4f}, Precision={precision:.4f}")
    assert abs(f1 - 0.0) < 1e-6, f"F1 mismatch: {f1}"
    assert abs(iou - 0.0) < 1e-6, f"IoU mismatch: {iou}"
    print("✓ PASS")


def test_known_tp_fp_fn_tn():
    """
    Custom case: TP=10, FP=5, FN=5, TN=80
    F1 = 2*TP / (2*TP + FP + FN) = 2*10 / (20 + 5 + 5) = 20/30 = 0.6667
    IoU = TP / (TP + FP + FN) = 10 / (10 + 5 + 5) = 10/20 = 0.5
    Precision = TP / (TP + FP) = 10 / 15 = 0.6667
    Recall = TP / (TP + FN) = 10 / 15 = 0.6667
    """
    print("\n=== Test 4: Known TP/FP/FN/TN (10, 5, 5, 80) ===")
    
    # Manually construct a 100-pixel image with known counts
    preds = torch.zeros(1, 1, 10, 10, dtype=torch.int32)
    masks = torch.zeros(1, 1, 10, 10, dtype=torch.int32)
    
    # TP: 10 pixels (both pred and mask are 1)
    preds[0, 0, :5, 0] = 1
    masks[0, 0, :5, 0] = 1
    preds[0, 0, 5:10, 0] = 1
    masks[0, 0, 5:10, 0] = 1
    
    # FP: 5 pixels (pred=1, mask=0)
    preds[0, 0, :5, 1] = 1
    masks[0, 0, :5, 1] = 0
    
    # FN: 5 pixels (pred=0, mask=1)
    preds[0, 0, 5:10, 1] = 0
    masks[0, 0, 5:10, 1] = 1
    
    # TN: remaining 80 pixels (both pred and mask are 0)
    # (already zeros from initialization)
    
    tp, fp, fn, tn = smp.metrics.get_stats(preds, masks.long(), mode='binary')
    iou = smp.metrics.iou_score(tp, fp, fn, tn).mean().item()
    f1 = smp.metrics.f1_score(tp, fp, fn, tn).mean().item()
    precision = smp.metrics.precision(tp, fp, fn, tn).mean().item()
    recall = smp.metrics.recall(tp, fp, fn, tn).mean().item()
    
    expected_f1 = 20 / 30  # 0.6667
    expected_iou = 10 / 20  # 0.5
    expected_precision = 10 / 15  # 0.6667
    expected_recall = 10 / 15  # 0.6667
    
    print(f"TP={tp.item()}, FP={fp.item()}, FN={fn.item()}, TN={tn.item()}")
    print(f"Expected: F1={expected_f1:.4f}, IoU={expected_iou:.4f}, Precision={expected_precision:.4f}, Recall={expected_recall:.4f}")
    print(f"Actual:   F1={f1:.4f}, IoU={iou:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
    
    assert abs(f1 - expected_f1) < 1e-4, f"F1 mismatch: expected {expected_f1}, got {f1}"
    assert abs(iou - expected_iou) < 1e-4, f"IoU mismatch: expected {expected_iou}, got {iou}"
    assert abs(precision - expected_precision) < 1e-4, f"Precision mismatch: expected {expected_precision}, got {precision}"
    assert abs(recall - expected_recall) < 1e-4, f"Recall mismatch: expected {expected_recall}, got {recall}"
    print("✓ PASS")


def test_f1_vs_iou_relationship():
    """
    Verify that F1 and IoU follow the mathematical relationship:
    F1 = 2*IoU / (1 + IoU)
    OR equivalently:
    IoU = F1 / (2 - F1)
    """
    print("\n=== Test 5: F1/IoU Mathematical Relationship ===")
    
    # Random predictions
    torch.manual_seed(42)
    preds = (torch.rand(1, 1, 32, 32) > 0.5).int()
    masks = (torch.rand(1, 1, 32, 32) > 0.5).int()
    
    tp, fp, fn, tn = smp.metrics.get_stats(preds, masks.long(), mode='binary')
    iou = smp.metrics.iou_score(tp, fp, fn, tn).mean().item()
    f1 = smp.metrics.f1_score(tp, fp, fn, tn).mean().item()
    
    # Verify F1 = 2*IoU / (1 + IoU)
    expected_f1_from_iou = 2 * iou / (1 + iou)
    
    print(f"IoU={iou:.6f}, F1={f1:.6f}")
    print(f"Expected F1 from IoU relationship: {expected_f1_from_iou:.6f}")
    print(f"Difference: {abs(f1 - expected_f1_from_iou):.8f}")
    
    assert abs(f1 - expected_f1_from_iou) < 1e-5, f"F1/IoU relationship violated: F1={f1}, expected={expected_f1_from_iou}"
    print("✓ PASS - F1 and IoU follow correct mathematical relationship")


if __name__ == "__main__":
    print("=" * 70)
    print("SYNTHETIC METRIC VALIDATION TEST")
    print("=" * 70)
    
    try:
        test_perfect_prediction()
        test_all_false_negatives()
        test_all_false_positives()
        test_known_tp_fp_fn_tn()
        test_f1_vs_iou_relationship()
        
        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED - Metrics are computed correctly")
        print("=" * 70)
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        raise
