import os
import sys
import zipfile
import xml.etree.ElementTree as ET
import csv
from datetime import datetime, timedelta

def excel_date_to_datetime(serial):
    if serial is None or serial == '':
        return None
    try:
        f = float(serial)
        # Excel 1900 date system (with the 1900 leap year bug):
        # 1 = 1900-01-01. Day 60 is 1900-02-29 (doesn't exist in reality).
        # For dates > 60:
        base = datetime(1899, 12, 30)
        dt = base + timedelta(days=f)
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        return str(serial)

def parse_xlsx_to_csv(input_path, output_path):
    print(f"Parsing {input_path} -> {output_path}...")
    with zipfile.ZipFile(input_path) as z:
        sst = []
        if 'xl/sharedStrings.xml' in z.namelist():
            tree = ET.parse(z.open('xl/sharedStrings.xml'))
            sst = [''.join(node.itertext()) for node in tree.findall('.//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}si')]
        
        tree = ET.parse(z.open('xl/worksheets/sheet1.xml'))
        rows = tree.findall('.//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}row')
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f_out:
            writer = csv.writer(f_out)
            header = None
            col_keys = []
            
            for i, row in enumerate(rows):
                row_dict = {}
                for cell in row.findall('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}c'):
                    r = cell.attrib.get('r')
                    col = ''.join(filter(str.isalpha, r))
                    t = cell.attrib.get('t')
                    v_el = cell.find('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v')
                    val = ''
                    if v_el is not None:
                        val = v_el.text
                        if t == 's':
                            val = sst[int(val)]
                    row_dict[col] = val
                
                if i == 0:
                    sorted_cols = sorted(row_dict.keys(), key=lambda x: (len(x), x))
                    header = [row_dict[c] for c in sorted_cols]
                    col_keys = sorted_cols
                    writer.writerow(header)
                else:
                    row_data = []
                    for col_idx, c in enumerate(col_keys):
                        val = row_dict.get(c, '')
                        col_name = header[col_idx]
                        if 'ts' in col_name.lower():
                            val = excel_date_to_datetime(val)
                        row_data.append(val)
                    writer.writerow(row_data)
                
                if (i + 1) % 50000 == 0:
                    print(f"  Processed {i+1} rows...")
    print(f"Finished {output_path} ({len(rows)} rows)")

if __name__ == '__main__':
    os.makedirs('processed_data', exist_ok=True)
    parse_xlsx_to_csv('abitur_stage1/Зарегистрированы (2014-2023).xlsx', 'processed_data/registered.csv')
    parse_xlsx_to_csv('abitur_stage1/Включены в КС (2014-2023).xlsx', 'processed_data/included_ks.csv')
    parse_xlsx_to_csv('abitur_stage1/Статусы заявлений (2014-2023).xlsx', 'processed_data/statuses.csv')

