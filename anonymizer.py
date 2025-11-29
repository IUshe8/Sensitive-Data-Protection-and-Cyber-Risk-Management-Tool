import pandas as pd
import hashlib
import re
from config_healthcare_v9 import ANONYMIZATION_CONFIG_HEALTHCARE

class Anonymizer:
    def __init__(self, config):
        self.config = config
        self.qi_columns = []
        self.sensitive_columns = []
        self.target_k = 5 # Define our target k

    def _calculate_los_binned(self, df, output_col, col_admission, col_discharge, bins, labels, na_default_label):
        try:
            adm = pd.to_datetime(df[col_admission], errors='coerce')
            dis = pd.to_datetime(df[col_discharge], errors='coerce')
            los = (dis - adm).dt.days
            binned_los = pd.cut(los, bins=bins, labels=labels, right=False)
            filled_los = binned_los.fillna(na_default_label)
            return filled_los.astype(str)
        except Exception as e:
            print(f"Error in LoS calculation: {e}")
            return pd.Series(index=df.index, name=output_col, dtype=str).fillna(na_default_label)

    def _hash_value(self, value):
        if pd.isna(value): return value
        return hashlib.sha256(str(value).encode()).hexdigest()

    def _generalize_bin(self, series, bins, labels):
        numeric_series = pd.to_numeric(series, errors='coerce')
        return pd.cut(numeric_series, bins=bins, labels=labels, right=False)

    def _generalize_simple_map(self, series, mapping, default_val='Other'):
        return series.astype(str).apply(lambda x: mapping.get(x, default_val))

    def _redact_regex(self, series, patterns): # (Not used in currently but might be useful)
        def redact(cell_value):
            for pat in patterns:
                cell_value = re.sub(pat, 'REDACTED', str(cell_value))
            return cell_value
        return series.apply(redact)

    def anonymize(self, df):
        """Main function to apply all rules from the config."""
        output_df = df.copy()
        
        for col_name, rule in self.config.items():
            if rule['type'] == 'calculate_los_binned':
                print(f"Calculating rule for: {col_name}")
                output_df[col_name] = self._calculate_los_binned(
                    output_df,
                    output_col=col_name,
                    col_admission=rule['col_admission'],
                    col_discharge=rule['col_discharge'],
                    bins=rule['bins'],
                    labels=rule['labels'],
                    na_default_label=rule['na_default_label']
                )
                self.qi_columns.append(col_name)

        for col_name, rule in self.config.items():
            rule_type = rule['type']
            if rule_type == 'calculate_los_binned': continue
                
            if col_name not in output_df.columns:
                print(f"Warning: Column '{col_name}' from config not in data. Skipping.")
                continue

            if rule_type == 'remove':
                output_df = output_df.drop(columns=[col_name])
            elif rule_type == 'generalize_bin':
                output_df[col_name] = self._generalize_bin(output_df[col_name], rule['bins'], rule['labels'])
                self.qi_columns.append(col_name)
            elif rule_type == 'generalize_simple_map':
                output_df[col_name] = self._generalize_simple_map(output_df[col_name], rule['mapping'], rule.get('default', 'Other'))
                self.qi_columns.append(col_name)
            elif rule_type == 'qi_exact':
                self.qi_columns.append(col_name)
            elif rule_type == 'sensitive':
                self.sensitive_columns.append(col_name)
        
        final_cols = self.qi_columns + self.sensitive_columns
        return output_df[final_cols]

    def augment_to_k(self, df_anonymized):
        """
        Finds groups < k and pads them with synthetic data.
        """
        print("\n--- Starting Augmentation Phase ---")
        
        df_qi_grouped = df_anonymized[self.qi_columns].astype(str).fillna('NA_VALUE')
        k_counts = df_qi_grouped.groupby(self.qi_columns).size()
        
        failing_groups = k_counts[k_counts < self.target_k]
        
        if failing_groups.empty:
            print("No augmentation needed. All groups meet k-target.")
            return df_anonymized
            
        print(f"Found {len(failing_groups)} groups to augment.")
        
        # Create a "pool" of real sensitive data to sample from
        # We sample *combinations* to keep data realistic
        sensitive_data_pool = df_anonymized[self.sensitive_columns].dropna()
        
        new_records = []
        for group_tuple, k_value in failing_groups.items():
            n_to_add = self.target_k - k_value
            
            # Create the QI part of the new records
            new_qi_data = dict(zip(self.qi_columns, group_tuple))
            
            # Sample sensitive data from our pool
            new_sensitive_data = sensitive_data_pool.sample(n=n_to_add, replace=True)
            
            # Combine QIs and Sensitive data for each new record
            for i in range(n_to_add):
                new_record = new_qi_data.copy()
                new_record.update(new_sensitive_data.iloc[i].to_dict())
                new_records.append(new_record)

        if not new_records:
            return df_anonymized

        print(f"Generated {len(new_records)} synthetic records to meet k={self.target_k}.")
        
        # Add the new records to the dataframe
        df_new = pd.DataFrame(new_records)
        df_augmented = pd.concat([df_anonymized, df_new], ignore_index=True)
        
        return df_augmented

if __name__ == "__main__":
    input_filename = "healthcare_dataset.csv"
    output_filename = "anonymized_augmented_dataset.csv" # <-- NEW FILENAME
    
    print(f"--- Loading data from '{input_filename}' ---")
    df_raw = pd.read_csv(input_filename)

    # --- Step 1: Anonymize ---
    print("--- Starting Anonymization Process (V9 Rules) ---")
    tool = Anonymizer(config=ANONYMIZATION_CONFIG_HEALTHCARE)
    df_anonymized = tool.anonymize(df_raw)

    # --- Step 2: Augment ---
    df_augmented = tool.augment_to_k(df_anonymized)

    print("\n--- ANONYMIZED & AUGMENTED DATA (First 5 Rows) ---")
    print(df_augmented.head())

    df_augmented.to_csv(output_filename, index=False)
    print(f"\nProcess complete. Data saved to '{output_filename}'")