# Data Cleaning Summary Report

- **Initial Rows:** 1067371
- **Rows after Deduplication:** 1033036
- **Rows after Dropping Null Customer/Description:** 797885
- **Rows after Removing Cancellations:** 779495
- **Rows after Removing Negative Quantity/Price:** 779425
- **Final Valid Rows:** 779425
- **Total Rows Removed:** 287946

## Dataset Preview
Columns: ['invoice', 'stockcode', 'description', 'quantity', 'invoicedate', 'price', 'customer_id', 'country', 'revenue']
Data Types:
invoice                object
stockcode              object
description            object
quantity                int64
invoicedate    datetime64[ns]
price                 float64
customer_id            object
country                object
revenue               float64
