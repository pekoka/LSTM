import sys
import csv
import datetime
import argparse

##設定値
#元データのCSVファイル名　元データのCSVファイルは「https://api.bitcoincharts.com/v1/csv/」で取得
input_file_name = "data20180922.csv"
#作成する4本値のCSVファイル名
output_file_name = "trance20180922.csv"
#集計を開始する日付＋時刻
kijyun_date = datetime.datetime.strptime('20170704 17:15:00', '%Y%m%d %H:%M:%S')
#分足の単位。例えば5分足なら「minutes=5」
kizami_date = datetime.timedelta(minutes=5)
#kizami_date = datetime.timedelta(minutes=3)
#kizami_date = datetime.timedelta(minutes=1)

# build up command line argment parser
parser = argparse.ArgumentParser(
            usage='OHLCV data Converter',
            description='description',
            epilog='',
            add_help=True, )
parser.add_argument('-i', '--infile',  help='input csv data file')
parser.add_argument('-o', '--outfile', help='output csv data file')
parser.add_argument('-t', '--interval', help='OHLCV interval(60, 180, 300)',type=int)

# 引数を解析する
args = parser.parse_args()
if args.infile:
    input_file_name = args.infile
if args.outfile:
    output_file_name = args.outfile
if args.interval:
    kizami_date = args.interval

if __name__ == '__main__':
    #CSV読み込み
    csv_file = open(input_file_name, "r", encoding="ms932", errors="", newline="" )
    f = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="", quotechar='"', skipinitialspace=True)

    datalist = []#計算結果の格納
    price_start = 0#始値
    price_max = 0#高値
    price_min = 0#安値
    price_end = 0#終値
    trading_volume = 0#売買出来高
    for row in f:
        if (datetime.datetime.fromtimestamp(int(row[0])) < kijyun_date + kizami_date):
            if price_start == 0:
                price_start = int(row[1].replace('.000000000000',''))
                price_max = int(row[1].replace('.000000000000',''))
                price_min = int(row[1].replace('.000000000000',''))
                price_end = int(row[1].replace('.000000000000',''))
                trading_volume = float(row[2])
            else:
                if price_max < int(row[1].replace('.000000000000','')):
                    price_max = int(row[1].replace('.000000000000',''))
                if price_min > int(row[1].replace('.000000000000','')):
                    price_min = int(row[1].replace('.000000000000',''))
                trading_volume += float(row[2])
        else:
            if price_start > 0:
                price_end = int(row[1].replace('.000000000000',''))
            datalist_new = int(row[0]), price_start, price_max, price_min, price_end, trading_volume
            if price_start == 0 and price_max == 0 and price_min == 0 and price_end == 0 and trading_volume == 0:
                print("No Traidng")
            else:
                datalist.append(datalist_new)
            kijyun_date = kijyun_date + kizami_date
            price_start = 0
            price_max = 0
            price_min = 0
            price_end = 0
            trading_volume = 0

    #CSVファイルに出力する
    csv_file = open(output_file_name, 'w', encoding='UTF-8')
    csv_writer = csv.writer(csv_file, lineterminator='\n')
    for j in range(len(datalist)):
        csv_writer.writerow(datalist[j])
    csv_file.close()
    print("convert done.")
