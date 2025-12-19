-- Fabric notebook source

-- METADATA ********************

-- META {
-- META   "kernel_info": {
-- META     "name": "sqldatawarehouse"
-- META   },
-- META   "dependencies": {
-- META     "warehouse": {
-- META       "default_warehouse": "15fefd71-838d-bf05-4328-7fd0beaf69fa",
-- META       "known_warehouses": [
-- META         {
-- META           "id": "15fefd71-838d-bf05-4328-7fd0beaf69fa",
-- META           "type": "Datawarehouse"
-- META         }
-- META       ]
-- META     }
-- META   }
-- META }

-- CELL ********************

-- Welcome to your new notebook
-- Type here in the cell editor to add code!


-- METADATA ********************

-- META {
-- META   "language": "sql",
-- META   "language_group": "sqldatawarehouse"
-- META }

-- CELL ********************

SELECT TOP (100) [CityKey],
			[WWICityID],
			[City],
			[StateProvince],
			[Country],
			[Continent],
			[ValidTo],
			[LineageKey],
			[SalesTerritory],
			[Region],
			[Subregion],
			[Location],
			[LatestRecordedPopulation],
			[ValidFrom]
FROM [part5_Wide World Importers].[dbo].[dimension_city]

-- METADATA ********************

-- META {
-- META   "language": "sql",
-- META   "language_group": "sqldatawarehouse"
-- META }
