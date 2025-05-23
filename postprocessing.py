import numpy as np
from collections import defaultdict


# 1. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
def ctr(b):       
    x1,y1,x2,y2=b;  return (x1+x2)/2, (y1+y2)/2
def w(b):         
    x1,_,x2,_=b;    return x2-x1
def h(b): 
    _,y1,_,y2=b;    return y2-y1
def horizontally_close(b1,b2,thr=0.6):
    x11,_,x12,_=b1; x21,_,x22,_=b2
    overlap=max(0, min(x12,x22)-max(x11,x21))
    return overlap > thr*min(w(b1),w(b2))

# классификация некоторых «базовых» символов
SUPPORTS_LIMITS = {"\\int", "\\sum", "\\prod", "\\lim"}


# 2. ПОСТ-ПРОЦЕССИНГ YOLO: ПОИСК ОТНОШЕНИЙ
def build_relation_graph(boxes, x_gap=0.6, y_gap=0.35):
    """
    Возвращает граф, где для каждой вершины-символа хранится список потомков
    вида {"type": "sup"/"sub"/"over"/"under"/"radicand", "child": j}.
    """
    n=len(boxes)
    graph=[[] for _ in range(n)]
    
    # Предварительно сортируем слева-направо
    order=sorted(range(n), key=lambda i: ctr(boxes[i]["bbox"])[0])
    
    for i in order:
        bi=boxes[i]["bbox"]
        xi, yi = ctr(bi)
        hi       = h(bi)
        is_limit = boxes[i]["label"] in SUPPORTS_LIMITS

        for j in range(n):
            if i==j: continue
            bj = boxes[j]["bbox"]
            if not horizontally_close(bi,bj,thr=x_gap): continue 
            xj,yj = ctr(bj)

            
            dy = (yi - yj)/hi

            if dy > y_gap:  
                if is_limit:
                    rel="sup"        # для int/lim 
                else:
                    rel="sup"        #верхний индекс
            elif dy < -y_gap: # ниже
                if is_limit:
                    rel="sub"
                else:
                    rel="sub"
            else:
                continue
            graph[i].append({"type": rel, "child": j})
    
    for i,b in enumerate(boxes):
        if b["label"] in {"√", "sqrt"}:
            cand=find_radicand(i, boxes)
            if cand is not None:
                graph[i].append({"type":"radicand","child":cand})
    return graph

def find_radicand(i, boxes):
    """
    Ищем ближайший bbox справа на том же уровне.
    """
    xi,yi = ctr(boxes[i]["bbox"])
    cand=None; dx_min=1e9
    for j,b in enumerate(boxes):
        if j==i: continue
        xj,yj=ctr(b["bbox"])
        if yj > yi-0.3*h(b["bbox"]) and yj < yi+0.3*h(b["bbox"]) and xj>xi:
            dx=xj-xi
            if dx<dx_min:
                dx_min=dx; cand=j
    return cand

def to_latex(boxes, graph):
    visited=set()
    def dfs(i):
        if i in visited: return ""
        visited.add(i)
        node = boxes[i]["label"]
        # соберём связи
        sup = [e["child"] for e in graph[i] if e["type"]=="sup"]
        sub = [e["child"] for e in graph[i] if e["type"]=="sub"]
        over= [e["child"] for e in graph[i] if e["type"]=="over"]
        under=[e["child"] for e in graph[i] if e["type"]=="under"]
        rad  = [e["child"] for e in graph[i] if e["type"]=="radicand"]

        result=node
        if rad:
            result = f"\\sqrt{{{dfs(rad[0])}}}"
        if over and under:
            result = f"\\frac{{{dfs(over[0])}}}{{{dfs(under[0])}}}"
        if sup:
            result += f"^{{{''.join(dfs(k) for k in sup)}}}"
        if sub:
            result += f"_{{{''.join(dfs(k) for k in sub)}}}"
        return result

    # проходим узлы слева-направо
    latex=[]
    for i in sorted(range(len(boxes)), key=lambda k: ctr(boxes[k]["bbox"])[0]):
        if i not in visited:
            latex.append(dfs(i))
    return ''.join(latex)