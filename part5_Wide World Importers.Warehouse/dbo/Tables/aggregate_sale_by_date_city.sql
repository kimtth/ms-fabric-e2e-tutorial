CREATE TABLE [dbo].[aggregate_sale_by_date_city] (

	[Date] datetime2(6) NULL, 
	[City] varchar(8000) NULL, 
	[StateProvince] varchar(8000) NULL, 
	[SalesTerritory] varchar(8000) NULL, 
	[SumOfTotalExcludingTax] decimal(38,2) NULL, 
	[SumOfTaxAmount] decimal(38,6) NULL, 
	[SumOfTotalIncludingTax] decimal(38,6) NULL, 
	[SumOfProfit] decimal(38,2) NULL
);