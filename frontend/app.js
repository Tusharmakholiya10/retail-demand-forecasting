const API_BASE_URL = "http://127.0.0.1:8000";


/* =========================
   DOM ELEMENTS
========================= */

const apiStatus =
    document.getElementById("api-status");

const storeCount =
    document.getElementById("store-count");

const familyCount =
    document.getElementById("family-count");

const bestModel =
    document.getElementById("best-model");

const apiKpi =
    document.getElementById("api-kpi");

const metricsHeader =
    document.getElementById("metrics-header");

const metricsBody =
    document.getElementById("metrics-body");

const storesList =
    document.getElementById("stores-list");

const familiesList =
    document.getElementById("families-list");

const systemMessage =
    document.getElementById("system-message");


const storeFilter =
    document.getElementById("store-filter");

const familyFilter =
    document.getElementById("family-filter");

const dateFrom =
    document.getElementById("date-from");

const dateTo =
    document.getElementById("date-to");

const applyFiltersButton =
    document.getElementById("apply-filters");

const resetFiltersButton =
    document.getElementById("reset-filters");

const filterMessage =
    document.getElementById("filter-message");


const forecastRecords =
    document.getElementById("forecast-records");

const totalActual =
    document.getElementById("total-actual");

const totalPredicted =
    document.getElementById("total-predicted");

const avgActual =
    document.getElementById("avg-actual");

const avgPredicted =
    document.getElementById("avg-predicted");

const forecastTableHead =
    document.getElementById("forecast-table-head");

const forecastTableBody =
    document.getElementById("forecast-table-body");


/* =========================
   API HELPER
========================= */

async function fetchAPI(endpoint) {

    const response =
        await fetch(`${API_BASE_URL}${endpoint}`);


    if (!response.ok) {

        throw new Error(
            `API request failed: ${response.status}`
        );

    }


    return await response.json();

}


/* =========================
   HEALTH
========================= */

async function loadHealth() {

    try {

        const data =
            await fetchAPI("/health");


        apiStatus.textContent =
            "API Online";


        apiKpi.textContent =
            "Online";


        systemMessage.textContent =
            `${data.service} is running successfully.`;


    } catch (error) {

        console.error(
            "Health check failed:",
            error
        );


        apiStatus.textContent =
            "API Offline";


        apiKpi.textContent =
            "Offline";


        systemMessage.textContent =
            "Unable to connect to FastAPI.";

    }

}


/* =========================
   LOAD STORES
========================= */

async function loadStores() {

    try {

        const data =
            await fetchAPI("/stores");


        storeCount.textContent =
            data.count;


        storeFilter.innerHTML =
            `<option value="">All Stores</option>`;


        storesList.innerHTML =
            "";


        data.stores.forEach(store => {


            const option =
                document.createElement("option");


            option.value =
                store;


            option.textContent =
                `Store ${store}`;


            storeFilter.appendChild(
                option
            );


            const tag =
                document.createElement("span");


            tag.className =
                "tag";


            tag.textContent =
                `Store ${store}`;


            storesList.appendChild(
                tag
            );

        });


    } catch (error) {

        console.error(
            "Failed to load stores:",
            error
        );


        storeCount.textContent =
            "--";

    }

}


/* =========================
   LOAD FAMILIES
========================= */

async function loadFamilies() {

    try {

        const data =
            await fetchAPI("/families");


        familyCount.textContent =
            data.count;


        familyFilter.innerHTML =
            `<option value="">All Families</option>`;


        familiesList.innerHTML =
            "";


        data.families.forEach(family => {


            const option =
                document.createElement("option");


            option.value =
                family;


            option.textContent =
                family;


            familyFilter.appendChild(
                option
            );


            const tag =
                document.createElement("span");


            tag.className =
                "tag";


            tag.textContent =
                family;


            familiesList.appendChild(
                tag
            );

        });


    } catch (error) {

        console.error(
            "Failed to load families:",
            error
        );


        familyCount.textContent =
            "--";

    }

}


/* =========================
   LOAD MODEL DATA
========================= */

async function loadMetrics() {

    try {

        const data =
            await fetchAPI("/models");


        if (
            !data.data ||
            data.data.length === 0
        ) {

            metricsBody.innerHTML =
                `<tr>
                    <td colspan="10">
                        No model data available.
                    </td>
                </tr>`;

            return;

        }


        const columns =
            Object.keys(
                data.data[0]
            );


        metricsHeader.innerHTML =
            "";


        columns.forEach(column => {

            const th =
                document.createElement("th");


            th.textContent =
                formatColumnName(column);


            metricsHeader.appendChild(
                th
            );

        });


        metricsBody.innerHTML =
            "";


        data.data.forEach(row => {

            const tr =
                document.createElement("tr");


            columns.forEach(column => {

                const td =
                    document.createElement("td");


                td.textContent =
                    formatValue(
                        row[column]
                    );


                tr.appendChild(td);

            });


            metricsBody.appendChild(tr);

        });


        const modelColumn =
            findColumn(
                columns,
                [
                    "model",
                    "model_name",
                    "algorithm"
                ]
            );


        if (modelColumn) {

            bestModel.textContent =
                data.data[0][modelColumn];

        }


    } catch (error) {

        console.error(
            "Failed to load model data:",
            error
        );

    }

}


/* =========================
   LOAD FORECAST
========================= */

async function loadForecast() {

    try {

        filterMessage.textContent =
            "Loading forecast data...";


        const params =
            new URLSearchParams();


        const store =
            storeFilter.value;


        const family =
            familyFilter.value;


        const from =
            dateFrom.value;


        const to =
            dateTo.value;


        if (store) {

            params.append(
                "store",
                store
            );

        }


        if (family) {

            params.append(
                "family",
                family
            );

        }


        if (from) {

            params.append(
                "date_from",
                from
            );

        }


        if (to) {

            params.append(
                "date_to",
                to
            );

        }


        /*
         * Request enough records for
         * the selected filters.
         */

        params.append(
            "limit",
            "50000"
        );


        const endpoint =
            `/forecast?${params.toString()}`;


        const data =
            await fetchAPI(endpoint);


        filterMessage.textContent =
            `${data.total_matches.toLocaleString()} matching records found. Showing ${data.count.toLocaleString()}.`;


        renderForecastSummary(
            data.data
        );


        renderForecastChart(
            data.data
        );


        renderForecastTable(
            data.data
        );


    } catch (error) {

        console.error(
            "Failed to load forecast:",
            error
        );


        filterMessage.textContent =
            "Unable to load forecast data.";


        clearForecast();

    }

}


/* =========================
   FORECAST SUMMARY
========================= */

function renderForecastSummary(data) {

    forecastRecords.textContent =
        data.length.toLocaleString();


    const actualColumn =
        findActualColumn(data);


    const predictionColumn =
        findPredictionColumn(data);


    if (
        !actualColumn ||
        !predictionColumn
    ) {

        totalActual.textContent =
            "--";

        totalPredicted.textContent =
            "--";

        avgActual.textContent =
            "--";

        avgPredicted.textContent =
            "--";

        return;

    }


    const actualValues =
        getNumericValues(
            data,
            actualColumn
        );


    const predictedValues =
        getNumericValues(
            data,
            predictionColumn
        );


    const actualTotal =
        actualValues.reduce(
            (sum, value) =>
                sum + value,
            0
        );


    const predictedTotal =
        predictedValues.reduce(
            (sum, value) =>
                sum + value,
            0
        );


    const actualAverage =
        actualValues.length
            ? actualTotal /
              actualValues.length
            : 0;


    const predictedAverage =
        predictedValues.length
            ? predictedTotal /
              predictedValues.length
            : 0;


    totalActual.textContent =
        formatNumber(
            actualTotal
        );


    totalPredicted.textContent =
        formatNumber(
            predictedTotal
        );


    avgActual.textContent =
        formatNumber(
            actualAverage
        );


    avgPredicted.textContent =
        formatNumber(
            predictedAverage
        );

}


/* =========================
   FORECAST CHART
========================= */

function renderForecastChart(data) {

    const chart =
        document.getElementById(
            "forecast-chart"
        );


    if (!data || data.length === 0) {

        chart.innerHTML =
            "<p>No forecast data available for the selected filters.</p>";

        return;

    }


    const dateColumn =
        findDateColumn(data);


    const actualColumn =
        findActualColumn(data);


    const predictionColumn =
        findPredictionColumn(data);


    if (
        !dateColumn ||
        !actualColumn ||
        !predictionColumn
    ) {

        chart.innerHTML =
            `
            <p>
                Unable to identify date, actual,
                or prediction columns.
            </p>
            `;

        console.error(
            "Forecast columns:",
            data.length
                ? Object.keys(data[0])
                : []
        );

        return;

    }


    /*
     * Aggregate by date.
     *
     * This is important because the
     * forecast dataset contains many
     * store/product-family records
     * per date.
     */

    const aggregated =
        {};


    data.forEach(row => {

        const date =
            row[dateColumn];


        const actual =
            Number(
                row[actualColumn]
            );


        const predicted =
            Number(
                row[predictionColumn]
            );


        if (!date) {

            return;

        }


        if (!aggregated[date]) {

            aggregated[date] = {

                actual: 0,

                predicted: 0

            };

        }


        if (
            Number.isFinite(actual)
        ) {

            aggregated[date].actual +=
                actual;

        }


        if (
            Number.isFinite(predicted)
        ) {

            aggregated[date].predicted +=
                predicted;

        }

    });


    const dates =
        Object.keys(
            aggregated
        ).sort();


    const actualValues =
        dates.map(
            date =>
                aggregated[date].actual
        );


    const predictedValues =
        dates.map(
            date =>
                aggregated[date].predicted
        );


    const traces = [

        {

            x: dates,

            y: actualValues,

            mode: "lines",

            name: "Actual Demand",

            line: {
                width: 2
            }

        },

        {

            x: dates,

            y: predictedValues,

            mode: "lines",

            name: "Predicted Demand",

            line: {
                width: 2,
                dash: "dash"
            }

        }

    ];


    const layout = {

        title: "",

        xaxis: {

            title: "Date",

            type: "date"

        },

        yaxis: {

            title: "Demand"

        },

        hovermode: "x unified",

        margin: {

            l: 65,

            r: 30,

            t: 20,

            b: 60

        },

        legend: {

            orientation: "h",

            y: 1.08,

            x: 0

        },

        paper_bgcolor:
            "white",

        plot_bgcolor:
            "white"

    };


    Plotly.newPlot(
        chart,
        traces,
        layout,
        {
            responsive: true,
            displaylogo: false
        }
    );

}


/* =========================
   FORECAST TABLE
========================= */

function renderForecastTable(data) {

    forecastTableHead.innerHTML =
        "";


    forecastTableBody.innerHTML =
        "";


    if (
        !data ||
        data.length === 0
    ) {

        forecastTableBody.innerHTML =
            `
            <tr>
                <td>
                    No forecast data available.
                </td>
            </tr>
            `;

        return;

    }


    const columns =
        Object.keys(
            data[0]
        );


    /*
     * Display only a reasonable
     * number of columns.
     */

    const visibleColumns =
        columns.slice(
            0,
            10
        );


    const headerRow =
        document.createElement(
            "tr"
        );


    visibleColumns.forEach(
        column => {

            const th =
                document.createElement(
                    "th"
                );


            th.textContent =
                formatColumnName(
                    column
                );


            headerRow.appendChild(
                th
            );

        }
    );


    forecastTableHead.appendChild(
        headerRow
    );


    /*
     * Limit visible rows so
     * browser doesn't become slow.
     */

    const visibleRows =
        data.slice(
            0,
            100
        );


    visibleRows.forEach(row => {

        const tr =
            document.createElement(
                "tr"
            );


        visibleColumns.forEach(
            column => {

                const td =
                    document.createElement(
                        "td"
                    );


                td.textContent =
                    formatValue(
                        row[column]
                    );


                tr.appendChild(td);

            }
        );


        forecastTableBody.appendChild(
            tr
        );

    });

}


/* =========================
   CLEAR FORECAST
========================= */

function clearForecast() {

    forecastRecords.textContent =
        "--";

    totalActual.textContent =
        "--";

    totalPredicted.textContent =
        "--";

    avgActual.textContent =
        "--";

    avgPredicted.textContent =
        "--";


    forecastTableHead.innerHTML =
        "";


    forecastTableBody.innerHTML =
        "";


    const chart =
        document.getElementById(
            "forecast-chart"
        );


    chart.innerHTML =
        "<p>No forecast data available.</p>";

}


/* =========================
   COLUMN DETECTION
========================= */

function findDateColumn(data) {

    if (!data.length) {
        return null;
    }


    const columns =
        Object.keys(data[0]);


    return findColumn(
        columns,
        [
            "date"
        ]
    );

}


function findActualColumn(data) {

    if (!data.length) {
        return null;
    }


    const columns =
        Object.keys(data[0]);


    return findColumn(
        columns,
        [
            "actual",
            "actual_sales",
            "sales",
            "y_true",
            "target"
        ]
    );

}


function findPredictionColumn(data) {

    if (!data.length) {
        return null;
    }


    const columns =
        Object.keys(data[0]);


    return findColumn(
        columns,
        [
            "predicted",
            "prediction",
            "predicted_sales",
            "forecast",
            "y_pred"
        ]
    );

}


function findColumn(
    columns,
    possibleNames
) {

    /*
     * Exact match first.
     */

    for (
        const name of possibleNames
    ) {

        const exact =
            columns.find(
                column =>
                    column.toLowerCase()
                    === name.toLowerCase()
            );


        if (exact) {

            return exact;

        }

    }


    /*
     * Then partial match.
     */

    return columns.find(
        column => {

            const lower =
                column.toLowerCase();


            return possibleNames.some(
                name =>
                    lower.includes(
                        name.toLowerCase()
                    )
            );

        }
    );

}


/* =========================
   NUMERIC HELPERS
========================= */

function getNumericValues(
    data,
    column
) {

    return data
        .map(
            row =>
                Number(row[column])
        )
        .filter(
            value =>
                Number.isFinite(value)
        );

}


function formatNumber(value) {

    if (
        !Number.isFinite(value)
    ) {

        return "--";

    }


    return value.toLocaleString(
        undefined,
        {
            maximumFractionDigits: 2
        }
    );

}


function formatValue(value) {

    if (
        value === null ||
        value === undefined
    ) {

        return "-";

    }


    if (
        typeof value === "number"
    ) {

        return value.toLocaleString(
            undefined,
            {
                maximumFractionDigits: 4
            }
        );

    }


    return value;

}


function formatColumnName(
    column
) {

    return column
        .replaceAll("_", " ")
        .replace(
            /\b\w/g,
            char =>
                char.toUpperCase()
        );

}


/* =========================
   RESET FILTERS
========================= */

function resetFilters() {

    storeFilter.value =
        "";

    familyFilter.value =
        "";

    dateFrom.value =
        "";

    dateTo.value =
        "";


    loadForecast();

}


/* =========================
   EVENT LISTENERS
========================= */

applyFiltersButton
    .addEventListener(
        "click",
        loadForecast
    );


resetFiltersButton
    .addEventListener(
        "click",
        resetFilters
    );


/* =========================
   INITIALIZE
========================= */

async function initializeDashboard() {

    console.log(
        "Initializing dashboard..."
    );


    await Promise.all([

        loadHealth(),

        loadStores(),

        loadFamilies(),

        loadMetrics()

    ]);


    /*
     * Load initial forecast
     * after metadata is available.
     */

    await loadForecast();


    console.log(
        "Dashboard initialization complete."
    );

}


initializeDashboard();