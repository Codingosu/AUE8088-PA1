from torchmetrics import Metric
import torch
import src.config as cfg # new

# [TODO] Implement this!
class MyF1Score(Metric):
    def __init__(self):
        super().__init__()
        self.num_classes = cfg.NUM_CLASSES

        # dim 0 : predict / dim 1 : truth
        self.add_state("Ev_Matrix", default=torch.zeros(self.num_classes, self.num_classes), dist_reduce_fx="sum")
        
    def update(self, preds, target):
      
        pred_labels = torch.argmax(preds, dim=1)
        for predict, truth in zip(pred_labels, target):
            self.Ev_Matrix[predict, truth] += 1

    def compute(self):
       
        TP = self.Ev_Matrix.diag()                        
        FP = self.Ev_Matrix.sum(dim=1) - TP               
        FN = self.Ev_Matrix.sum(dim=0) - TP            

        precision = TP/(TP + FP)
        recall = TP/(TP + FN)
        f1_all = 2*precision*recall/(precision + recall)

        ## for check
        ALL = TP + FN

        cols = [("label",   5), ("TP",       8), ("FN",       8), ("TP+FN",    8), ("F1Score",  8)]

        header = " | ".join(f"{name:>{w}}" for name, w in cols)
        sep = "".join("|" if ch == "|" else "-" for ch in header)
        
        print(sep)
        print(header)
        print(sep)
        for i in range(self.num_classes):
            tp   = int(TP[i].item())
            fn   = int(FN[i].item())
            all_ = int(ALL[i].item())
            f1   =        f1_all[i].item()
            row = " | ".join([
                f"{i:>{cols[0][1]}}",
                f"{tp:>{cols[1][1]}}",
                f"{fn:>{cols[2][1]}}",
                f"{all_:>{cols[3][1]}}",
                f"{f1:>{cols[4][1]}.3f}" 
            ])
            print(row)
        print(sep)
        
        ''' # ----- [추가] F1 낮은 순으로 정렬 출력 -----
        # print("\n[Sorted by F1 Score (Lowest First)]")
        # print(sep)
        # print(header)
        # print(sep)
        # # (label, TP, FN, TP+FN, F1) tuple list 만들기
        # rows = []
        # for i in range(self.num_classes):
        #     tp   = int(TP[i].item())
        #     fn   = int(FN[i].item())
        #     all_ = int(ALL[i].item())
        #     f1   =        f1_all[i].item()
        #     rows.append((i, tp, fn, all_, f1))

        # # F1 기준으로 오름차순 정렬
        # rows_sorted = sorted(rows, key=lambda x: x[4])  # f1 기준

        # # 출력
        # for i, tp, fn, all_, f1 in rows_sorted:
        #     row = " | ".join([
        #         f"{i:>{cols[0][1]}}",
        #         f"{tp:>{cols[1][1]}}",
        #         f"{fn:>{cols[2][1]}}",
        #         f"{all_:>{cols[3][1]}}",
        #         f"{f1:>{cols[4][1]}.3f}" 
        #     ])
        #     print(row)
        # print(sep)
        '''
         # ----- [추가] FP-FN 기하평균 기준 정렬 -----
        print("\n[Sorted by Geometric Mean of (FP * FN)]")
        
        cols = [("label",   5), ("TP",       8), ("F1Score",  8), ("GeoMean", 8)]
        header = " | ".join(f"{name:>{w}}" for name, w in cols)
        sep = "".join("|" if ch == "|" else "-" for ch in header)

        print(sep)
        print(header)
        print(sep)
        
        geo_rows = []
        for i in range(self.num_classes):
            tp   = int(TP[i].item())
            f1   = f1_all[i].item()
            
            fp   = int(FP[i].item())
            fn   = int(FN[i].item())
            gmean = (fp * fn) ** 0.5
            geo_rows.append((i, tp, f1, gmean))

        # 기하평균 기준 내림차순 정렬
        geo_rows_sorted = sorted(geo_rows, key=lambda x: -x[3])

        for i, tp, f1, gmean in geo_rows_sorted:
            row = " | ".join([
                f"{i:>{cols[0][1]}}",
                f"{tp:>{cols[1][1]}}",
                f"{f1:>{cols[2][1]}.3f}",
                f"{gmean:>{cols[3][1]}.3f}",
            ])
            print(row)
        print(sep)
        
        return f1_all

    ## for check
    def show(self):
        """
        show [10x10] block of Ev_matrix
        """
        s = 10
        sub_Matrix = self.Ev_Matrix[:s, :s]
        
        # visualization
        header = " " * 8 + "".join(f"{j:>6}" for j in range(s))
        print("Ev_Matrix sub-block (predicted × truth)")
        print(header)
        print("-" * len(header))

        for i in range(s):
            row = sub_Matrix[i]
            line = f"pred {i:2d} |" + "".join(f"{int(v):6d}" for v in row)
            print(line)
    
    ## for check
    def show_all(self):
        """
        show [10x10] block of Ev_matrix
        """
        s = 10
        sub_Matrix = self.Ev_Matrix[:s, :s]
        
        # visualization
        header = " " * 8 + "".join(f"{j:>6}" for j in range(s))
        print("Ev_Matrix sub-block (predicted × truth)")
        print(header)
        print("-" * len(header))

        for i in range(s):
            row = sub_Matrix[i]
            line = f"pred {i:2d} |" + "".join(f"{int(v):6d}" for v in row)
            print(line)
 
class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # [TODO] The preMYds (B x C tensor), so take argmax to get index with highest confidence
        # B : # batch, C : # class
        pred_labels = preds.argmax(dim=1)
        
        # [TODO] check if preds and target have equal shape
        # print()
        # print(preds.shape)
        # print(pred_labels.shape)
        # print(target.shape)
        # if target.shape == pred_labels.shape:
        #     print("now, comment these out")

        # [TODO] Count the number of correct prediction
        correct = (pred_labels == target).sum()

        # Accumulate to self.correct
        self.correct += correct

        # Count the number of elements in target
        self.total += target.numel()

        ## for check
        # print("--- Debug MyAccuracy.update ---")
        # print("preds       :", preds.tolist())      # raw output
        # print("target      :", target.item())       # ground‐truth label
        # print("pred_labels :", pred_labels.item())  # label predicted
        # print("correct     :", correct.item())   # correct
        # print("------------------------------")
        
    def compute(self):
        return self.correct.float() / self.total.float()
