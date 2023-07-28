# UDF guide
Here we instruct how to use our provided UDFs.
## Preparation
1. Set database connect information in ```/UDF_dependencies/mbi/dataset.py```, ```DADP_Initialize.sql``` and  ```DADP_Update.sql```.
2. Copy all folders in  ```UDF_dependencies``` to your Python site-package folder, like
```
/usr/local/python3/lib/python3.8/site-packages/
```
or
```
~/.local/python3/lib/python3.8/site-packages/
```

3. Create tables with structures shown below.

|Name|Column name|Data Source|Note
|----|----|----|----|
|```dataset```|Attributes of ```dataset```|```dataset```.csv| Partial dataset for initialize, not full dataset
|```dataset```_domain|DOMAIN, SIZE|```dataset```-domain.json|
|```dataset```_synth_domain|DOMAIN, SIZE|```dataset```-domain.json|Same as ```dataset```_domain
|prefer_cliques|0, 1|None|Select prefer cliques by your self, and insert it to this table.

4. Create ```plpython3u``` language/extension in PostgreSQL by using
```
CREATE EXTENSION plpython3u;
```
## Usage
1. Import ```DADP_Initialize.sql``` and ```DADP_Update.sql``` to PostgreSQL with ```psql``` or simply paste the content of files which mentioned.
2. Run ```DADP_Initialize()``` with
```
SELECT dadp_init(dataset, budget);
```
3. Insert more records to table ```dataset```, then run ```DADP_Update()``` with
```
SELECT dadp_update(dataset, budget);
```
