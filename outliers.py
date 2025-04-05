import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/average-monthly-surface-temperature.csv")


def generate_histograms(ctr, hdf):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(hdf['Average surface temperature daily'], bins=30, color='blue')
    plt.title(f'Daily average temperatures in {ctr}')
    plt.xlabel('Temperature')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(hdf['Average surface temperature monthly'], bins=30, color='blue')
    plt.title(f'Monthly average temperature in {ctr}')
    plt.xlabel('Temperature')
    plt.ylabel('Frequency')

    plt.tight_layout()


def generate_boxplot(ct, bdf):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.boxplot(bdf['Average surface temperature daily'])
    plt.title(f'Boxplot - Daily temperature - {ct}')
    plt.xlabel('Temperature')

    plt.subplot(1, 2, 2)
    plt.boxplot(bdf['Average surface temperature monthly'])
    plt.title(f'Boxplot - Monthly temperature - {ct}')
    plt.xlabel('Temperature')


def find_outliers(column, l=1.5):
    q1 = column.quantile(0.25)
    q3 = column.quantile(0.75)
    iqr = q3 - q1
    inferior_limit = q1 - l * iqr
    superior_limit = q3 + l * iqr
    outliers = column[(column < inferior_limit) | (column > superior_limit)]
    return outliers


def identify_outliers_superior_limit(column, l=3):
    q1 = column.quantile(0.25)
    q3 = column.quantile(0.75)
    iqr = q3 - q1
    superior_limit = q3 + l * iqr
    outliers = column[(column > superior_limit)]
    return outliers


def process_date_info(no_outlier_df):
    no_outlier_df['data_datetime'] = pd.to_datetime(no_outlier_df['Day'])
    no_outlier_df['year_month'] = no_outlier_df['data_datetime'].dt.to_period('M')

    initial_month = no_outlier_df['year_month'].min()
    final_month = no_outlier_df['year_month'].max()
    print(f"First month: {initial_month}")
    print(f"Last month: {final_month}")

    all_months = pd.period_range(start=initial_month, end=final_month, freq='M')
    existing_months = no_outlier_df['year_month'].unique()
    missing_months = all_months.difference(existing_months)

    print(f"Number of missing months: {len(missing_months)}")
    if not missing_months.empty:
        print("Missing months: ")
        for month in missing_months:
            print(str(month))

    monthly_data = no_outlier_df.groupby('year_month').mean(numeric_only=True)
    monthly_data = monthly_data.reindex(all_months)
    monthly_data['Average surface temperature daily'] = monthly_data['Average surface temperature daily'].interpolate(
        method='linear')
    monthly_data.reset_index(inplace=True)
    monthly_data.rename(columns={'index': 'year_month'}, inplace=True)
    monthly_data['data_datetime'] = monthly_data['year_month'].dt.to_timestamp()
    print(monthly_data)

    return monthly_data


def process_country_data(country, df):
    print(f'\nCountry: {country}')
    filtered_df = df[df['Entity'] == country]

    generate_histograms(country, filtered_df)

    daily_outliers = find_outliers(filtered_df['Average surface temperature daily'])
    monthly_outliers = find_outliers(filtered_df['Average surface temperature monthly'])
    if not daily_outliers.empty:
        print(f"Daily outliers: \n{daily_outliers}")
    if not monthly_outliers.empty:
        print(f"Monthly outliers: \n{monthly_outliers}")

    generate_boxplot(country, filtered_df)

    no_outlier_df = filtered_df[~filtered_df['Average surface temperature daily'].isin(
        identify_outliers_superior_limit(filtered_df['Average surface temperature daily'])
    )].copy()
    generate_histograms(country, no_outlier_df)

    # Process date information and compute missing months
    monthly_data = process_date_info(no_outlier_df)

    # Save processed monthly data to CSV
    output_filename = f"data/{country.replace(' ', '_').lower()}_tempered.csv"
    monthly_data.to_csv(output_filename, index=False)
    print(f"Monthly aggregated data saved to {output_filename}")

    plt.show()
    plt.close()


def main():
    countries = ['Chad', 'Faroe Islands', 'Jamaica']
    for country in countries:
        process_country_data(country, df)


if __name__ == "__main__":
    main()
