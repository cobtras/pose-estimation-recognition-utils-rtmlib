# tracking.py
import numpy as np
from typing import List


class PersonTracker:
    def __init__(self, iou_threshold: float = 0.0, keypoint_weight: float = 0.5, debug: bool = False):
        self.iou_threshold=iou_threshold
        self.keypoint_weight=keypoint_weight
        self.debug=debug
        self.next_id=0
        self.prev_tracks=[]  # Liste von Dicts: {'id': int, 'bbox': [x1,y1,x2,y2], 'keypoints': np.ndarray}
        self.frame_idx=0

    def bbox_iou(self, box1: List[float], box2: List[float]) -> float:
        """Berechnet Intersection over Union für zwei Bounding Boxen [x1,y1,x2,y2]."""
        x1=max(box1[0], box2[0])
        y1=max(box1[1], box2[1])
        x2=min(box1[2], box2[2])
        y2=min(box1[3], box2[3])
        inter=max(0, x2 - x1) * max(0, y2 - y1)
        area1=(box1[2] - box1[0]) * (box1[3] - box1[1])
        area2=(box2[2] - box2[0]) * (box2[3] - box2[1])
        union=area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def match_persons(self, new_bboxes: List[np.ndarray], new_keypoints: List[np.ndarray]):
        """
        Ordnet neue Personen alten IDs zu.

        Args:
            new_bboxes: Liste von Bounding Boxen [x1,y1,x2,y2] (als np.ndarray oder Liste)
            new_keypoints: Liste von Keypoint-Arrays der Form (num_keypoints, 2)

        Returns:
            person_ids: Liste der IDs für jede neue Person (Länge = len(new_bboxes))
            max_id: maximale ID nach diesem Frame (kann gestiegen sein)
        """
        n_new=len(new_bboxes)
        n_old=len(self.prev_tracks)
        person_ids=[-1] * n_new
        
        self.frame_idx += 1
        if self.debug:
            print(f"\n--- Frame {self.frame_idx} Tracking Debug ---")
            print(f"Alte Tracks: {[t['id'] for t in self.prev_tracks]}")
            print(f"Neue Detections: {n_new}")

        if n_old == 0:
            # Keine alten Tracks: alle neuen Personen bekommen neue IDs
            for i in range(n_new):
                person_ids[i]=self.next_id
                if self.debug:
                    print(f"Neu: Person {i} -> ID {self.next_id}")
                self.next_id+=1
        else:
            # Kostenmatrix berechnen
            cost_matrix=np.ones((n_new, n_old))
            for i in range(n_new):
                for j in range(n_old):
                    iou=self.bbox_iou(new_bboxes[i], self.prev_tracks[j]['bbox'])
                    
                    if iou < self.iou_threshold:
                        if self.debug:
                            print(f"Verwerfe [Neu {i} -> Alt {self.prev_tracks[j]['id']}]: IoU {iou:.4f} < {self.iou_threshold}")
                        continue
                        
                    iou_cost=1 - iou

                    kp_new=new_keypoints[i]
                    kp_old=self.prev_tracks[j]['keypoints']
                    valid=np.all(kp_new != 0, axis=1) & np.all(kp_old != 0, axis=1)
                    if np.sum(valid) > 0:
                        distances=np.linalg.norm(kp_new[valid] - kp_old[valid], axis=1)
                        kp_cost=np.mean(distances) / 100.0  # Normierung
                    else:
                        kp_cost=1.0

                    cost_matrix[i, j]=(1 - self.keypoint_weight) * iou_cost + self.keypoint_weight * kp_cost
                    if self.debug:
                        print(f"Match-Kandidat [Neu {i} -> Alt {self.prev_tracks[j]['id']}]: IoU {iou:.4f}, IoU-Cost {iou_cost:.4f}, KP-Cost {kp_cost:.4f}, Gesamt-Kosten {cost_matrix[i, j]:.4f}")

            # Greedy Matching
            matched_new=set()
            matched_old=set()
            for i in range(n_new):
                if len(matched_old) == n_old:
                    break
                best_j=np.argmin(cost_matrix[i])
                if cost_matrix[i, best_j] < 1.0 and best_j not in matched_old:
                    person_ids[i]=self.prev_tracks[best_j]['id']
                    matched_new.add(i)
                    matched_old.add(best_j)
                    if self.debug:
                        print(f"MATCH: Neu {i} -> Alt {person_ids[i]} (Kosten: {cost_matrix[i, best_j]:.4f})")

            # Neue Personen (unmatched) erhalten neue IDs
            for i in range(n_new):
                if person_ids[i] == -1:
                    person_ids[i]=self.next_id
                    if self.debug:
                        print(f"Unmatched Neu {i} -> Neue ID {self.next_id}")
                    self.next_id+=1

        # Nächsten Tracking-Status speichern
        self.prev_tracks=[]
        for i in range(n_new):
            self.prev_tracks.append({
                'id': person_ids[i],
                'bbox': new_bboxes[i],
                'keypoints': new_keypoints[i]
            })

        max_id=self.next_id - 1
        return person_ids, max_id

    def reset(self):
        """Setzt den Tracker zurück (für ein neues Video)."""
        self.next_id=0
        self.prev_tracks=[]