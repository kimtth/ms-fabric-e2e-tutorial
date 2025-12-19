[DefaultOutputDestinationSettings = [DestinationDefinition = [Kind = "Reference", QueryName = "DefaultDestination", IsNewTarget = true], UpdateMethod = [Kind = "Replace"], DestinationTypeSettings = [Kind = "Table"]], StagingDefinition = [Kind = "FastCopy"]]
section Section1;
shared DefaultDestination = Lakehouse.Contents([EnableFolding = false]){[workspaceId = "b93996d5-c37f-44ef-bdf6-3cb0e5794a19"]}[Data]{[lakehouseId = "31efa6fb-1485-4b40-bc03-551461492aa0"]}[Data];
[DataDestinations = {[Definition = [Kind = "Reference", QueryName = "dimension_customer_DataDestination", IsNewTarget = true], Settings = [Kind = "Automatic", TypeSettings = [Kind = "Table"]]]}]
shared dimension_customer = let
  Source = Csv.Document(Web.Contents("https://raw.githubusercontent.com/microsoft/fabric-samples/35fd4439f02a045050fe6e951db259fc6c9392ba/docs-samples/data-engineering/dimension_customer.csv"), [Delimiter = ",", Columns = 11, Encoding = 65001, QuoteStyle = QuoteStyle.None]),
  #"Promoted headers" = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),
  #"Changed column type" = Table.TransformColumnTypes(#"Promoted headers", {{"CustomerKey", Int64.Type}, {"WWICustomerID", Int64.Type}, {"Customer", type text}, {"BillToCustomer", type text}, {"Category", type text}, {"BuyingGroup", type text}, {"PrimaryContact", type text}, {"PostalCode", type text}, {"ValidFrom", type datetime}, {"ValidTo", type datetime}, {"LineageKey", Int64.Type}})
in
  #"Changed column type";
shared dimension_customer_DataDestination = let
  Pattern = Lakehouse.Contents([CreateNavigationProperties = false, EnableFolding = false]),
  Navigation_1 = Pattern{[workspaceId = "b93996d5-c37f-44ef-bdf6-3cb0e5794a19"]}[Data],
  Navigation_2 = Navigation_1{[lakehouseId = "31efa6fb-1485-4b40-bc03-551461492aa0"]}[Data],
  TableNavigation = Navigation_2{[Id = "dimension_customer", ItemKind = "Table"]}?[Data]?
in
  TableNavigation;
