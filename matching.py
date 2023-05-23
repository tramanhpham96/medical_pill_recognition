from Levenshtein import distance as lev_dist 
import re

def find_match_drugname(pres_name, lib_names):
    score = []
    for lib_name in lib_names:
        score.append(lev_dist(pres_name, lib_name))
    return lib_names[score.index(min(score))]

def find_pres_id_qty_list(pres_drugnames,pres_quantities,mapping_id_name):
    lib_drugnames = [name for name in mapping_id_name['Drugname']]
    lib_drugnames = list(dict.fromkeys(lib_drugnames))
    pres_list, id_list, qty_list = [], [], []
    for pres_name,pres_qty in zip(pres_drugnames,pres_quantities):
        pres_lib_drugname = find_match_drugname(pres_name, lib_drugnames)
        df_id_pres_name = mapping_id_name[mapping_id_name['Drugname']==pres_lib_drugname]
        ids_pres = df_id_pres_name['classID'].to_list()
        pres_list = pres_list + len(ids_pres)*[pres_lib_drugname]
        qty_list = qty_list + len(ids_pres)*[pres_qty]
        id_list = id_list + ids_pres
    
    return pres_list, id_list, qty_list 

def ID_drugname_match(pill_output, pres_drugnames,pres_quantities, mapping_id_name):
    yolo_ids = pill_output['cls']
    yolo_boxes = pill_output['boxes']
    pres_list, id_list, qty_list = find_pres_id_qty_list(pres_drugnames,pres_quantities,mapping_id_name)
    # table_dict = {'drugname':[],'qty':[],'box':[]}
    table_list =[]
    drugin = []
    new_yolo_ids, yolo_drugnames = [], []
    for yolo_id,yolo_box in zip(yolo_ids,yolo_boxes):
        if yolo_id not in id_list:
            new_yolo_ids = new_yolo_ids + [107]
            table_list.append({'drugname': 'Not_in_prescription', 'qty':'0', 'box': yolo_box})
        else:
            new_yolo_ids = new_yolo_ids + [yolo_id]
            table_list.append({'drugname': pres_list[id_list.index(yolo_id)], 'qty':qty_list[id_list.index(yolo_id)], 'box': yolo_box})
            drugin.append(pres_list[id_list.index(yolo_id)])

    absent_drug = set(pres_list)-set(drugin)
    absence_dict = []
    for drug in absent_drug:
        absence_dict.append({'drugname': drug,'qty':qty_list[pres_list.index(drug)]})

    return table_list,absence_dict

def extract_table_dict(pill_output, pres_output,mapping_id_name):
    pres_drugnames = [re.sub(r'^[^A-Za-z]+', '', pres[0]) for pres in pres_output['text'][1]]
    pres_quantities = [re.findall('\d+',pres[1]) for pres in pres_output['text'][1]]
    # Match ocr_drugname to the drugname lib 
    table_dict,absence_dict = ID_drugname_match(pill_output, pres_drugnames,pres_quantities, mapping_id_name)
    return table_dict,absence_dict