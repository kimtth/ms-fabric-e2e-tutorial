# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "779b1198-31e0-4e1d-9544-3dff87b477bb",
# META       "default_lakehouse_name": "part5_Shortcut_Exercise",
# META       "default_lakehouse_workspace_id": "b93996d5-c37f-44ef-bdf6-3cb0e5794a19",
# META       "known_lakehouses": [
# META         {
# META           "id": "779b1198-31e0-4e1d-9544-3dff87b477bb"
# META         }
# META       ]
# META     }
# META   }
# META }

# CELL ********************

# Welcome to your new notebook
# Type here in the cell editor to add code!

df = spark.sql("SELECT * FROM part5_Shortcut_Exercise.dimension_customer LIMIT 1000")
display(df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
