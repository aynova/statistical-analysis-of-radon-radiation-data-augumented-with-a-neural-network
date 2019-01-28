# statistical-analysis-of-radon-radiation-data-augumented-with-a-neural-network
#the statistical analysis of radiation data and the training of a neural network for the filling in of gaps with missing data and the #prediction of the future radon concentration when given meteorological data
# the hourly data
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates


    def prob_convert(xi, logic=False):
        if logic is True:
            return xi * (1 - xi)
        return 1 / (1 + np.exp(-xi))


    print("extracting the file")

    excel_file = "C:/Users/Admin/Documents/Rn data March to Oct 2015b.xlsx"
    data = pd.read_excel(excel_file)

    times = pd.date_range('2015-03-31T02:49:00.000Z', '2015-10-20T14:48:00.000Z', freq='H')
    time_range = [str(x) + 'Z' for x in times]


    temperature = []
    relative_humidity = []
    pressure = []
    radon = []
    time = []
    count = 0
    for t in range(len(time_range)):

        if str(time_range[t]).split(":")[0] == str(data.iloc[count, 4]).split(":")[0]:
            temperature.append(data.iloc[count, 0])
            relative_humidity.append(data.iloc[count, 1])
            pressure.append(data.iloc[count, 2])
            radon.append(data.iloc[count, 3])
            time.append(time_range[t])
            count += 1
        else:
            temperature.append(0)
            relative_humidity.append(0)
            pressure.append(0)
            radon.append(0)
            time.append(time_range[t])

    print(count)
    print(temperature)
    print(relative_humidity)
    print(pressure)
    print(radon)
    print(time)

    plt.figure(0)
    locator = mdates.HourLocator()
    locator.MAXTICKS = 10000
    plt.title('temperature hourly')
    plt.plot(times, temperature)
    plt.setp(plt.gca().xaxis.get_majorticklabels(), 'rotation', 90)
    plt.gca().xaxis.set_major_locator(locator)


    plt.figure(1)
    locator = mdates.HourLocator()
    locator.MAXTICKS = 10000
    plt.title('Relative Humidity hourly')
    plt.plot(times, relative_humidity)
    plt.setp(plt.gca().xaxis.get_majorticklabels(), 'rotation', 90)
    plt.gca().xaxis.set_major_locator(locator)


    plt.figure(2)
    locator = mdates.HourLocator()
    locator.MAXTICKS = 10000
    plt.title('pressure hourly')
    plt.plot(times, pressure)
    plt.setp(plt.gca().xaxis.get_majorticklabels(), 'rotation', 90)
    plt.gca().xaxis.set_major_locator(locator)


    plt.figure(3)
    locator = mdates.HourLocator()
    locator.MAXTICKS = 10000
    plt.title('Radon hourly')
    plt.plot(times, radon)
    plt.setp(plt.gca().xaxis.get_majorticklabels(), 'rotation', 90)
    plt.gca().xaxis.set_major_locator(locator)


    plt.figure(4)
    locator = mdates.HourLocator()
    locator.MAXTICKS = 10000
    plt.title('combined graphs')
    plt.plot(times, temperature, 'green')
    plt.plot(times, relative_humidity, 'blue')
    plt.plot(times, pressure, 'red')
    plt.plot(times, radon, 'violet')
    plt.setp(plt.gca().xaxis.get_majorticklabels(), 'rotation', 90)
    plt.gca().xaxis.set_major_locator(locator)

    plt.show()

    n = len(data)

    in_put_s = dict()
    out_puts = dict()

    for i in range(3):
        in_put_s[i] = dict()
        in_put_s[i] = 10 * data.iloc[:, i]

    out_puts = 10 * data.iloc[:, 3]

    print("removing the decimals")

    for i in range(n):
        out_puts[i] = str(out_puts[i]).split(".")
        out_puts[i] = int(out_puts[i][0])
        for j in range(3):
            in_put_s[j][i] = str(in_put_s[j][i]).split(".")
            in_put_s[j][i] = int(in_put_s[j][i][0])

    print(out_puts, in_put_s[0], in_put_s[1], in_put_s[2])

    print("converting to binary")

    in_puts = dict()
    for i in range(n):
        out_puts[i] = str("{0:09b}".format(out_puts[i]))
        for j in range(3):
            in_put_s[j][i] = "{0:09b}".format(in_put_s[j][i])
        in_puts[i] = [in_put_s[0][i], in_put_s[1][i], in_put_s[2][i]]
        in_puts[i] = str("".join(in_puts[i]))

    print(out_puts, in_puts)

    print("creating the input and output matrices")

    x_inputs_DTRP = dict()
    y_outputs_RADON = dict()

    for i in range(n):
        x_inputs_DTRP[i] = np.array(np.zeros((27, 1)))
        y_outputs_RADON[i] = np.array(np.zeros((9, 1)))

    for i in range(n):
        for j in range(27):
            x_inputs_DTRP[i][j] = in_puts[i][j]

        for k in range(9):
            y_outputs_RADON[i][k] = out_puts[i][k]

    print(x_inputs_DTRP, y_outputs_RADON)


    np.random.seed(1)

    sy0 = 2 * np.random.random((54, 27)) - 1
    sy1 = 2 * np.random.random((9, 54)) - 1

    La0 = []
    La1 = []
    La2 = []
    La2E = []
    print("the batch learning begins")
    for j in range(60000):
        for i in range(n):
            La0 = x_inputs_DTRP[i]
            La1 = prob_convert(np.dot(sy0, La0))
            La2 = prob_convert(np.dot(sy1, La1))
            La2E = y_outputs_RADON[i] - La2

            La2D = La2E * prob_convert(La2, logic=True)
            La1E = sy1.T.dot(La2D)
            La1D = La1E * prob_convert(La1, logic=True)

            sy1 += La2D.dot(La1.T)
            sy0 += La1D.dot(La0.T)
        if j % 15000 == 0:
            print("error: ", np.mean(np.abs(La2E)))

    print("output after the training")

    La0 = x_inputs_DTRP[0]
    La1 = prob_convert(np.dot(sy0, La0))
    La2 = prob_convert(np.dot(sy1, La1))

    print(La2)


    def mat2list(mat):
        dim = mat.shape
        dic = []
        for i in range(dim[0]):
            for j in range(dim[1]):
                dic.append(mat[i][j])
        return dic


    def binary_approximator(d):
        b = []
        for i in range(len(d)):
            if 0.9 <= d[i] < 1:
                b.append(1)
            elif 0 < d[i] < 0.9:
                b.append(0)
            else:
                b.append('the value is not binary')
        c = ''.join(str(i) for i in b[:len(b)])
        d = int(c, 2)
        return d


    La0 = x_inputs_DTRP[1]
    La1 = prob_convert(np.dot(sy0, La0))
    La2 = prob_convert(np.dot(sy1, La1))

    output_dict = mat2list(La2)

    print(output_dict)

    output_data = binary_approximator(output_dict)

    print(output_data)



    #  the daily data

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates


    def prob_convert(xi, logic=False):
        if logic is True:
            return xi * (1 - xi)
        return 1 / (1 + np.exp(-xi))


    print("extracting the file")

    excel_file = "C:/Users/Admin/Documents/Rn data March to Oct 2015b.xlsx"
    data = pd.read_excel(excel_file)
    print(data)
    times = pd.date_range('2015-04-01T02:49:00.000Z', '2015-10-19T15:48:00.000Z', freq='d')
    time_daili = [str(x) + 'Z' for x in times]

    print(time_daili)
    le = len(data)
    count = 0

    Radon_daily = dict()
    Temperature_daily = dict()
    Rel_Humidity_daily = dict()
    Pressure_daily = dict()
    time_daily = dict()
    Radon_day = 0
    Temperature_day = 0
    Rel_Humidity_day = 0
    Pressure_day = 0
    time_day = []
    counter = 0
    a = str(data.iloc[0, 4])[9]

    for i in range(le):
        if str(data.iloc[i, 4])[9] == a:
            Radon_day += data.iloc[i, 3]
            Temperature_day += data.iloc[i, 0]
            Rel_Humidity_day += data.iloc[i, 1]
            Pressure_day += data.iloc[i, 2]
            time_day.append(data.iloc[i, 4])
            counter += 1
        if str(data.iloc[i, 4])[9] != a:
            a = str(data.iloc[i, 4])[9]
            Radon_daily[count] = Radon_day / counter
            Temperature_daily[count] = Temperature_day / counter
            Rel_Humidity_daily[count] = Rel_Humidity_day / counter
            Pressure_daily[count] = Pressure_day / counter
            if counter % 2 == 0:
                time_daily[count] = time_day[int(counter / 2)]
            else:
                time_daily[count] = time_day[int((counter / 2) - 1)]
            Radon_day = 0
            Temperature_day = 0
            Rel_Humidity_day = 0
            Pressure_day = 0
            time_day = []
            counter = 0
            count += 1

    daily_data_s = dict()
    for i in range(count):
        daily_data_s[i] = [Temperature_daily[i], Rel_Humidity_daily[i], Pressure_daily[i], Radon_daily[i], time_daily[i]]


    daily_data = pd.DataFrame([daily_data_s[k] for k in range(count)], columns=[
        'Temperature daily', 'Relative Humidity daily', 'Pressure daily', 'Radon daily', 'time daily'])

    print(daily_data)


    temperature = []
    relative_humidity = []
    pressure = []
    radon = []
    time = []
    count = 0
    for t in range(len(time_daili)):

        if time_daili[t].split(" ")[0] == str(time_daily[count]).split(" ")[0]:
            temperature.append(Temperature_daily[count])
            relative_humidity.append(Rel_Humidity_daily[count])
            pressure.append(Pressure_daily[count])
            radon.append(Radon_daily[count])
            time.append(time_daili[t])
            count += 1
        else:
            temperature.append(0)
            relative_humidity.append(0)
            pressure.append(0)
            radon.append(0)
            time.append(time_daili[t])

    print(count)
    print(temperature)
    print(relative_humidity)
    print(pressure)
    print(radon)
    print(time)

    plt.figure(0)
    locator = mdates.DayLocator()
    locator.MAXTICKS = 10000
    plt.title('temperature daily')
    plt.plot(times, temperature)
    plt.setp(plt.gca().xaxis.get_majorticklabels(), 'rotation', 90)
    plt.gca().xaxis.set_major_locator(locator)


    plt.figure(1)
    locator = mdates.DayLocator()
    locator.MAXTICKS = 10000
    plt.title('Relative Humidity daily')
    plt.plot(times, relative_humidity)
    plt.setp(plt.gca().xaxis.get_majorticklabels(), 'rotation', 90)
    plt.gca().xaxis.set_major_locator(locator)


    plt.figure(2)
    locator = mdates.DayLocator()
    locator.MAXTICKS = 10000
    plt.title('pressure daily')
    plt.plot(times, pressure)
    plt.setp(plt.gca().xaxis.get_majorticklabels(), 'rotation', 90)
    plt.gca().xaxis.set_major_locator(locator)


    plt.figure(3)
    locator = mdates.DayLocator()
    locator.MAXTICKS = 10000
    plt.title('Radon daily')
    plt.plot(times, radon)
    plt.setp(plt.gca().xaxis.get_majorticklabels(), 'rotation', 90)
    plt.gca().xaxis.set_major_locator(locator)


    plt.figure(4)
    locator = mdates.DayLocator()
    locator.MAXTICKS = 10000
    plt.title('combined daily graphs')
    plt.plot(times, temperature, 'green')
    plt.plot(times, relative_humidity, 'blue')
    plt.plot(times, pressure, 'red')
    plt.plot(times, radon, 'violet')
    plt.setp(plt.gca().xaxis.get_majorticklabels(), 'rotation', 90)
    plt.gca().xaxis.set_major_locator(locator)

    plt.show()

    n = len(daily_data)

    in_put_s = dict()
    out_puts = dict()

    for i in range(3):
        in_put_s[i] = dict()
        in_put_s[i] = 10 * daily_data.iloc[:, i]

    out_puts = 10 * daily_data.iloc[:, 3]

    print("removing the decimals")

    for i in range(n):
        out_puts[i] = str(out_puts[i]).split(".")
        out_puts[i] = int(out_puts[i][0])
        for j in range(3):
            in_put_s[j][i] = str(in_put_s[j][i]).split(".")
            in_put_s[j][i] = int(in_put_s[j][i][0])

    print(out_puts, in_put_s[0], in_put_s[1], in_put_s[2])

    print("converting to binary")

    in_puts = dict()
    for i in range(n):
        out_puts[i] = str("{0:09b}".format(out_puts[i]))
        for j in range(3):
            in_put_s[j][i] = "{0:09b}".format(in_put_s[j][i])
        in_puts[i] = [in_put_s[0][i], in_put_s[1][i], in_put_s[2][i]]
        in_puts[i] = str("".join(in_puts[i]))

    print(out_puts, in_puts)

    print("creating the input and output matrices")

    x_inputs_DTRP = dict()
    y_outputs_RADON = dict()

    for i in range(n):
        x_inputs_DTRP[i] = np.array(np.zeros((27, 1)))
        y_outputs_RADON[i] = np.array(np.zeros((9, 1)))

    for i in range(n):
        for j in range(27):
            x_inputs_DTRP[i][j] = in_puts[i][j]

        for k in range(9):
            y_outputs_RADON[i][k] = out_puts[i][k]

    print(x_inputs_DTRP, y_outputs_RADON)

    np.random.seed(1)

    sy0 = 2 * np.random.random((54, 27)) - 1
    sy1 = 2 * np.random.random((9, 54)) - 1

    La0 = []
    La1 = []
    La2 = []
    La2E = []
    print("the batch learning begins")
    for j in range(60000):
        for i in range(n):
            La0 = x_inputs_DTRP[i]
            La1 = prob_convert(np.dot(sy0, La0))
            La2 = prob_convert(np.dot(sy1, La1))
            La2E = y_outputs_RADON[i] - La2

            La2D = La2E * prob_convert(La2, logic=True)
            La1E = sy1.T.dot(La2D)
            La1D = La1E * prob_convert(La1, logic=True)

            sy1 += La2D.dot(La1.T)
            sy0 += La1D.dot(La0.T)
        if j % 15000 == 0:
            print("error: ", np.mean(np.abs(La2E)))

    print("output after the training")

    La0 = x_inputs_DTRP[0]
    La1 = prob_convert(np.dot(sy0, La0))
    La2 = prob_convert(np.dot(sy1, La1))

    print(La2)


    def mat2list(mat):
        dim = mat.shape
        dic = []
        for i in range(dim[0]):
            for j in range(dim[1]):
                dic.append(mat[i][j])
        return dic


    def binary_approximator(d):
        b = []
        for i in range(len(d)):
            if 0.9 <= d[i] < 1:
                b.append(1)
            elif 0 < d[i] < 0.9:
                b.append(0)
            else:
                b.append('the value is not binary')
        c = ''.join(str(i) for i in b[:len(b)])
        d = int(c, 2)
        return d


    La0 = x_inputs_DTRP[1]
    La1 = prob_convert(np.dot(sy0, La0))
    La2 = prob_convert(np.dot(sy1, La1))

    output_dict = mat2list(La2)

    print(output_dict)

    output_data = binary_approximator(output_dict)

    print(output_data)



    # the monthly data


    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates


    def prob_convert(xi, logic=False):
        if logic is True:
            return xi * (1 - xi)
        return 1 / (1 + np.exp(-xi))


    print("extracting the file")

    excel_file = "C:/Users/Admin/Documents/Rn data March to Oct 2015b.xlsx"
    data = pd.read_excel(excel_file)
    print(data)
    times = pd.date_range('2015-03-31T02:49:00.000Z', '2015-10-20T15:48:00.000Z', freq='m')
    time_monthli = [str(x) + 'Z' for x in times]

    print(time_monthli)
    le = len(data)
    count = 0

    Radon_monthly = dict()
    Temperature_monthly = dict()
    Rel_Humidity_monthly = dict()
    Pressure_monthly = dict()
    time_monthly = dict()
    Radon_month = 0
    Temperature_month = 0
    Rel_Humidity_month = 0
    Pressure_month = 0
    time_month = []
    counter = 0
    a = str(data.iloc[0, 4])[6]

    for i in range(le):
        if str(data.iloc[i, 4])[6] == a:
            Radon_month += data.iloc[i, 3]
            Temperature_month += data.iloc[i, 0]
            Rel_Humidity_month += data.iloc[i, 1]
            Pressure_month += data.iloc[i, 2]
            time_month.append(data.iloc[i, 4])
            counter += 1
        if str(data.iloc[i, 4])[6] != a:
            a = str(data.iloc[i, 4])[6]
            Radon_monthly[count] = Radon_month / counter
            Temperature_monthly[count] = Temperature_month / counter
            Rel_Humidity_monthly[count] = Rel_Humidity_month / counter
            Pressure_monthly[count] = Pressure_month / counter
            if counter % 2 == 0:
                time_monthly[count] = time_month[int(counter / 2)]
            else:
                time_monthly[count] = time_month[int((counter + 1) / 2)]
            Radon_month = 0
            Temperature_month = 0
            Rel_Humidity_month = 0
            Pressure_month = 0
            time_month = []
            counter = 0
            count += 1

    monthly_data_s = dict()
    for i in range(count):
        monthly_data_s[i] = [Temperature_monthly[i], Rel_Humidity_monthly[i], Pressure_monthly[i], Radon_monthly[i], time_monthly[i]]


    monthly_data = pd.DataFrame([monthly_data_s[k] for k in range(count)], columns=[
        'Temperature monthly', 'Relative Humidity monthly', 'Pressure monthly', 'Radon monthly', 'time monthly'])

    print(monthly_data)


    temperature = []
    relative_humidity = []
    pressure = []
    radon = []
    time = []
    count = 0
    for t in range(len(time_monthli)):

        if time_monthli[t].split("-")[1] == str(time_monthly[count]).split("-")[1]:
            temperature.append(Temperature_monthly[count])
            relative_humidity.append(Rel_Humidity_monthly[count])
            pressure.append(Pressure_monthly[count])
            radon.append(Radon_monthly[count])
            time.append(time_monthli[t])
            count += 1
        else:
            temperature.append(0)
            relative_humidity.append(0)
            pressure.append(0)
            radon.append(0)
            time.append(time_monthli[t])

    print(count)
    print(temperature)
    print(relative_humidity)
    print(pressure)
    print(radon)
    print(time)

    plt.figure(0)
    locator = mdates.MonthLocator()
    locator.MAXTICKS = 10000
    plt.title('temperature monthly')
    plt.plot(times, temperature)
    plt.setp(plt.gca().xaxis.get_majorticklabels(), 'rotation', 90)
    plt.gca().xaxis.set_major_locator(locator)


    plt.figure(1)
    locator = mdates.MonthLocator()
    locator.MAXTICKS = 10000
    plt.title('Relative Humidity monthly')
    plt.plot(times, relative_humidity)
    plt.setp(plt.gca().xaxis.get_majorticklabels(), 'rotation', 90)
    plt.gca().xaxis.set_major_locator(locator)


    plt.figure(2)
    locator = mdates.MonthLocator()
    locator.MAXTICKS = 10000
    plt.title('pressure monthly')
    plt.plot(times, pressure)
    plt.setp(plt.gca().xaxis.get_majorticklabels(), 'rotation', 90)
    plt.gca().xaxis.set_major_locator(locator)


    plt.figure(3)
    locator = mdates.MonthLocator()
    locator.MAXTICKS = 10000
    plt.title('Radon monthly')
    plt.plot(times, radon)
    plt.setp(plt.gca().xaxis.get_majorticklabels(), 'rotation', 90)
    plt.gca().xaxis.set_major_locator(locator)


    plt.figure(4)
    locator = mdates.MonthLocator()
    locator.MAXTICKS = 10000
    plt.title('combined monthly graphs')
    plt.plot(times, temperature, 'green')
    plt.plot(times, relative_humidity, 'blue')
    plt.plot(times, pressure, 'red')
    plt.plot(times, radon, 'violet')
    plt.setp(plt.gca().xaxis.get_majorticklabels(), 'rotation', 90)
    plt.gca().xaxis.set_major_locator(locator)

    plt.show()

    n = len(monthly_data)

    in_put_s = dict()
    out_puts = dict()

    for i in range(3):
        in_put_s[i] = dict()
        in_put_s[i] = 10 * monthly_data.iloc[:, i]

    out_puts = 10 * monthly_data.iloc[:, 3]

    print("removing the decimals")

    for i in range(n):
        out_puts[i] = str(out_puts[i]).split(".")
        out_puts[i] = int(out_puts[i][0])
        for j in range(3):
            in_put_s[j][i] = str(in_put_s[j][i]).split(".")
            in_put_s[j][i] = int(in_put_s[j][i][0])

    print(out_puts, in_put_s[0], in_put_s[1], in_put_s[2])

    print("converting to binary")

    in_puts = dict()
    for i in range(n):
        out_puts[i] = str("{0:09b}".format(out_puts[i]))
        for j in range(3):
            in_put_s[j][i] = "{0:09b}".format(in_put_s[j][i])
        in_puts[i] = [in_put_s[0][i], in_put_s[1][i], in_put_s[2][i]]
        in_puts[i] = str("".join(in_puts[i]))

    print(out_puts, in_puts)

    print("creating the input and output matrices")

    x_inputs_DTRP = dict()
    y_outputs_RADON = dict()

    for i in range(n):
        x_inputs_DTRP[i] = np.array(np.zeros((27, 1)))
        y_outputs_RADON[i] = np.array(np.zeros((9, 1)))

    for i in range(n):
        for j in range(27):
            x_inputs_DTRP[i][j] = in_puts[i][j]

        for k in range(9):
            y_outputs_RADON[i][k] = out_puts[i][k]

    print(x_inputs_DTRP, y_outputs_RADON)

    np.random.seed(1)

    sy0 = 2 * np.random.random((54, 27)) - 1
    sy1 = 2 * np.random.random((9, 54)) - 1

    La0 = []
    La1 = []
    La2 = []
    La2E = []
    print("the batch learning begins")
    for j in range(60000):
        for i in range(n):
            La0 = x_inputs_DTRP[i]
            La1 = prob_convert(np.dot(sy0, La0))
            La2 = prob_convert(np.dot(sy1, La1))
            La2E = y_outputs_RADON[i] - La2

            La2D = La2E * prob_convert(La2, logic=True)
            La1E = sy1.T.dot(La2D)
            La1D = La1E * prob_convert(La1, logic=True)

            sy1 += La2D.dot(La1.T)
            sy0 += La1D.dot(La0.T)
        if j % 15000 == 0:
            print("error: ", np.mean(np.abs(La2E)))

    print("output after the training")

    La0 = x_inputs_DTRP[0]
    La1 = prob_convert(np.dot(sy0, La0))
    La2 = prob_convert(np.dot(sy1, La1))

    print(La2)


    def mat2list(mat):
        dim = mat.shape
        dic = []
        for i in range(dim[0]):
            for j in range(dim[1]):
                dic.append(mat[i][j])
        return dic


    def binary_approximator(d):
        b = []
        for i in range(len(d)):
            if 0.9 <= d[i] < 1:
                b.append(1)
            elif 0 < d[i] < 0.9:
                b.append(0)
            else:
                b.append('the value is not binary')
        c = ''.join(str(i) for i in b[:len(b)])
        d = int(c, 2)
        return d


    La0 = x_inputs_DTRP[1]
    La1 = prob_convert(np.dot(sy0, La0))
    La2 = prob_convert(np.dot(sy1, La1))

    output_dict = mat2list(La2)

    print(output_dict)

    output_data = binary_approximator(output_dict)

    print(output_data)

