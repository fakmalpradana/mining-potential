# data/raw

Raw downloads live on the **shared Drive**, not in git (this folder is ignored except for this file).
Hyatma downloads everything by 19 Nov (task #3) and records it in `data_inventory.md`.
Copy the files you need from the Drive into this folder on your own machine.

All Tellus and GSNI data: UK Open Government Licence. No account is needed.

## Airborne geophysics (Tellus, 2005–2006)

```bash
wget https://opendatani.blob.core.windows.net/gsnigeospatial/Tellus_Magnetics_ASCII.zip
wget https://opendatani.blob.core.windows.net/gsnigeospatial/Tellus_Radiometrics_ASCII.zip
wget https://opendatani.blob.core.windows.net/gsnigeospatial/Tellus_Electromagnetics_ASCII.zip
```

## Soil geochemistry (Tellus Rural Soil Survey)

```bash
wget https://admin.opendatani.gov.uk/dataset/dc01f414-21e2-409c-8a8b-d34a67187ddd/resource/00961fff-d94d-452a-8654-9ddb998b717d/download/regionalsoilssaquaregiahdl.csv
wget https://admin.opendatani.gov.uk/dataset/dc01f414-21e2-409c-8a8b-d34a67187ddd/resource/e9a4568a-677d-419a-a2ce-31940900bbe9/download/regionalsoilssfireassayaupge.csv
wget https://admin.opendatani.gov.uk/dataset/dc01f414-21e2-409c-8a8b-d34a67187ddd/resource/1c35fb41-1c4e-4c33-956e-3b2e7850ee93/download/regionalsoilsaxrf.csv
```

Detection limits: methodology PDF on the [Rural Soil Survey resource page](https://www.opendatani.gov.uk/dataset/rural-soil-survey/resource/330e2a8f-db4e-42ec-9a66-145a5eb07c93).
Alternative source: [BGS Tellus Data Downloads](https://www2.bgs.ac.uk/gsni/tellus/data_licensing/index.html).

## Geology (GSNI 1:250,000)

```bash
wget https://admin.opendatani.gov.uk/dataset/7c00c1f5-6cd3-405d-b79e-61c19c4990b9/resource/07b32481-a218-4591-b245-bea5543a060f/download/ni_250k_bedrock_geology_polygons.gpkg
wget https://admin.opendatani.gov.uk/dataset/7c00c1f5-6cd3-405d-b79e-61c19c4990b9/resource/9380def2-3529-4d09-9eec-2d1f57241aa9/download/ni_250k_bedrock_geology_lines.gpkg
wget https://admin.opendatani.gov.uk/dataset/7c00c1f5-6cd3-405d-b79e-61c19c4990b9/resource/e490914d-9e13-4cc9-915b-3cac75643a95/download/ni_250k_superficial_geology_polygons.gpkg
```

## Gravity (EIGEN-6C4)

Generate manually at the [ICGEM Calculation Service](http://icgem.gfz-potsdam.de/calcgrid):
model EIGEN-6C4, functional "simple Bouguer gravity anomaly", 54.0–55.4° N, 8.2–5.4° W, grid step 0.01°, GeoTIFF.
Citation: Förste et al. (2014), https://doi.org/10.5880/ICGEM.2015.1
