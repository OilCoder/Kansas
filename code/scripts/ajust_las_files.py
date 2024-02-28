import os
import pandas as pd
import lasio
import zipfile
import json

def read_csv_data(csv_path):
    return pd.read_csv(csv_path)

def process_csv_data(csv_folder):
    files = os.listdir(csv_folder)
    wells_csv = next((f for f in files if f.startswith('Wells_')), None)
    tops_csv = next((f for f in files if f.startswith('Tops_')), None)
    logs_csv = next((f for f in files if f.startswith('Logs_')), None)
    LAS_csv = next((f for f in files if f.startswith('LAS_')), None)
    
    wells_data = read_csv_data(os.path.join(csv_folder, wells_csv)) if wells_csv else pd.DataFrame()
    tops_data = read_csv_data(os.path.join(csv_folder, tops_csv)) if tops_csv else pd.DataFrame()
    logs_data = read_csv_data(os.path.join(csv_folder, logs_csv)) if logs_csv else pd.DataFrame()
    LAS_data = read_csv_data(os.path.join(csv_folder, LAS_csv)) if LAS_csv else pd.DataFrame()

    return wells_data, tops_data, logs_data, LAS_data

def find_kid_for_las(las_file_name, LAS_data):
    kid_row = LAS_data[LAS_data['LASFILE'] == las_file_name]
    return kid_row['KID'].iloc[0] if not kid_row.empty else None

def update_las(las_file_path, well_data, log_data, top_data, LAS_data, destination_folder):
    error_log = []  # Lista para almacenar errores
    try:

        las = lasio.read(las_file_path)
        
        # Encuentra el KID basado en el nombre del archivo LAS.
        kid = find_kid_for_las(os.path.basename(las_file_path), LAS_data)

        if kid:
            filtered_well_data = well_data[well_data['KID'] == kid]
            filtered_log_data = log_data[log_data['KID'] == kid]
            filtered_top_data = top_data[top_data['KID'] == kid]

            # Asignaciones generales del pozo
            well_name = f"{filtered_well_data['LEASE_NAME'].iloc[0]} {filtered_well_data['WELL_NAME'].iloc[0]}"
            las.well['WELL'].value = well_name
            las.well['FLD'].value = filtered_well_data['FIELD_NAME'].iloc[0]
            # las.well['LOC'].value = filtered_log_data['LOCATION'].iloc[0]
            las.well['COMP'].value = filtered_well_data['ORIG_OPERATOR'].iloc[0]  # Operador original como compañía

            if 'LOC' in las.well:
                las.well['LOC'].value = filtered_log_data['LOCATION'].iloc[0]
            else:
                # Agregar el campo CNTY si es necesario
                las.well['LOC'] = lasio.HeaderItem(mnemonic='LOC', unit='', value=filtered_log_data['LOCATION'].iloc[0], descr='Location')

            # Información sobre la profundidad del pozo
            las.well['STRT'].value = filtered_log_data['TOP'].min() if not filtered_log_data.empty else None
            las.well['STOP'].value = filtered_log_data['BOTTOM'].max() if not filtered_log_data.empty else None
            #las.well['STEP'].value = None  # El paso podría ser calculado si es necesario y está disponible

            # Verificación y asignación condicional para CTRY
            if 'CTRY' in las.well:
                las.well['CTRY'].value = 'USA'
            else:
                # Opcionalmente, agregar el campo CTRY si es necesario para tu análisis
                las.well['CTRY'] = lasio.HeaderItem(mnemonic='CTRY', unit='', value='USA', descr='Country')
            # Verificación y asignación condicional para STAT
            if 'STAT' in las.well:
                las.well['STAT'].value = 'Kansas'
            else:
                # Opcionalmente, agregar el campo CTRY si es necesario para tu análisis
                las.well['STAT'] = lasio.HeaderItem(mnemonic='STAT', unit='', value='Kansas', descr='State')
            if 'PROV' in las.well:
                las.well['PROV'].value = filtered_well_data['TOWNSHIP'].iloc[0]
            else:
                # Agregar el campo PROV si es necesario
                las.well['PROV'] = lasio.HeaderItem(mnemonic='PROV', unit='', value=filtered_well_data['TOWNSHIP'].iloc[0], descr='Province')
            if 'CNTY' in las.well:
                las.well['CNTY'].value = filtered_well_data['COUNTY'].iloc[0]
            else:
                # Agregar el campo CNTY si es necesario
                las.well['CNTY'] = lasio.HeaderItem(mnemonic='CNTY', unit='', value=filtered_well_data['COUNTY'].iloc[0], descr='County')

            # asignacion de identificacion del pozo 
            if 'UWI' in las.well:
                las.well['UWI'].value = kid
            else:
                # Opcionalmente, agregar el campo CTRY si es necesario para tu análisis
                las.well['UWI'] = lasio.HeaderItem(mnemonic='UWI', unit='', descr='Unique Well Identifier')
            if 'API' in las.well:
                las.well['API'].value = filtered_well_data['API'].iloc[0]
            else:
                # Agregar el campo XCOORD si es necesario
                las.well['API'] = lasio.HeaderItem(mnemonic='API', unit='', value=filtered_well_data['API'].iloc[0], descr='API Number')

            # # Asignaciones específicas del pozo
            # las.well['SRVC'].value = filtered_log_data['LOGGER'].iloc[0]
            # las.well['DATE'].value = filtered_log_data['LOG_DATE'].iloc[0]
            
            if 'SRVC' in las.well:
                las.well['SRVC'].value = filtered_log_data['LOGGER'].iloc[0]
            else:
                # Agregar el campo CNTY si es necesario
                las.well['SRVC'] = lasio.HeaderItem(mnemonic='SRVC', unit='', value=filtered_log_data['LOGGER'].iloc[0], descr='Logger Service Company')
            if 'DATE' in las.well:
                las.well['DATE'].value = filtered_log_data['LOG_DATE'].iloc[0]
            else:
                # Agregar el campo CNTY si es necesario
                las.well['DATE'] = lasio.HeaderItem(mnemonic='DATE', unit='', value=filtered_log_data['LOG_DATE'].iloc[0], descr='Date logged')

            # Coordenadas geográficas y elevación
            # Verificación y asignación condicional para XCOORD
            if 'XCOORD' in las.well:
                las.well['XCOORD'].value = filtered_well_data['NAD27_LONGITUDE'].iloc[0]
            else:
                # Agregar el campo XCOORD si es necesario
                las.well['XCOORD'] = lasio.HeaderItem(mnemonic='XCOORD', unit='X', value=filtered_well_data['NAD27_LONGITUDE'].iloc[0], descr='X Coordinate')
            # Verificación y asignación condicional para YCOORD
            if 'YCOORD' in las.well:
                las.well['YCOORD'].value = filtered_well_data['NAD27_LATITUDE'].iloc[0]
            else:
                # Agregar el campo YCOORD si es necesario
                las.well['YCOORD'] = lasio.HeaderItem(mnemonic='YCOORD', unit='Y', value=filtered_well_data['NAD27_LATITUDE'].iloc[0], descr='Y Coordinate')

            

            
            # Otros campos importantes basados en los datos disponibles
            # Verificación y asignación condicional para PROV y CNTY
            # if 'TOWNSHIP' in filtered_well_data.columns and 'PROV' in las.well:
            #     las.well['PROV'].value = filtered_well_data['TOWNSHIP'].iloc[0]
            # if 'COUNTY' in filtered_well_data.columns:
            #     las.well['CNTY'].value = filtered_well_data['COUNTY'].iloc[0]


            # function to add other section with top_base_formation
            las = add_formation_info_to_las_other_section(las, filtered_top_data)

            # Guarda el archivo LAS actualizado en la carpeta de destino con un nuevo nombre
            new_file_name = f"{well_name.replace(' ', '_').replace('/', '_')}.las"
            destination_path = os.path.join(destination_folder, new_file_name)
            las.write(destination_path, version=2.0,fmt='%.4f',
                        column_fmt={0: '%.2f'}, mnemonics_header=True)
            
    except Exception as e:
        # Registro de errores
        error_info = {
            'file': las_file_path,
            'error': str(e),
            'message': 'Error desconocido durante la actualización del archivo LAS'
        }
        error_log.append(error_info)

    # Escribir la lista de errores en un archivo JSON al final de la función
    log_file_path = 'error_log.json'
    with open(log_file_path, 'w') as log_file:
        json.dump(error_log, log_file, indent=4)

def add_formation_info_to_las_other_section(las, top_data):

    # Preparar el encabezado CSV
    csv_header = "\tTOP, BASE, FORMATION\n"
    
    # Inicializar una lista para almacenar las líneas de datos de formación
    formation_lines = [csv_header]
    
    # Iterar a través del DataFrame de formaciones y formatear cada fila como CSV
    for index, row in top_data.iterrows():
        line = f"\t{row['TOP']}, {row['BASE']}, {row['FORMATION']}\n"
        formation_lines.append(line)
    
    # Unir todas las líneas en una sola cadena de texto
    formations_csv = "".join(formation_lines)
    
    # Añadir la cadena CSV formateada a la sección ~Other del archivo LAS
    if 'Other' in las.sections:
        las.sections['Other'] += formations_csv
    else:
        las.sections['Other'] = formations_csv
    
    return las

def unzip_files(source_folder, destination_folder):
    for zip_file in os.listdir(source_folder):
        if zip_file.endswith('.zip'):
            zip_file_path = os.path.join(source_folder, zip_file)
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(destination_folder)

import zipfile
import tempfile
import shutil

def process_las_files(source_folder, destination_folder, csv_folder):
    for field_folder_name in os.listdir(source_folder):
        field_source_folder = os.path.join(source_folder, field_folder_name)
        field_destination_folder = os.path.join(destination_folder, field_folder_name)
        field_csv_folder = os.path.join(csv_folder, field_folder_name)
        
        if os.path.isdir(field_source_folder) and os.path.isdir(field_csv_folder):
            wells_data, tops_data, logs_data, LAS_data = process_csv_data(field_csv_folder)
            
            if not os.path.isdir(field_destination_folder):
                os.makedirs(field_destination_folder)
            
            for file_name in os.listdir(field_source_folder):
                if file_name.lower().endswith('.zip'):
                    zip_file_path = os.path.join(field_source_folder, file_name)
                    temp_dir = tempfile.mkdtemp()  # Directorio temporal para extraer archivos
                    
                    print(zip_file_path)
                    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    for root, dirs, files in os.walk(temp_dir):
                        for file in files:
                            if file.lower().endswith('.las'):
                                las_file_path = os.path.join(root, file)
                                kid = find_kid_for_las(file, LAS_data)
                                if kid:
                                    filtered_well_data = wells_data[wells_data['KID'] == kid]
                                    filtered_top_data = tops_data[tops_data['KID'] == kid]
                                    filtered_log_data = logs_data[logs_data['KID'] == kid]
                                    if not filtered_well_data.empty and not filtered_top_data.empty and not filtered_log_data.empty:
                                        las = update_las(las_file_path, filtered_well_data, filtered_log_data, filtered_top_data, LAS_data, field_destination_folder)
                    # Eliminar directorio temporal después de procesar los archivos LAS
                    shutil.rmtree(temp_dir)