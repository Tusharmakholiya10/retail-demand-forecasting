const API_BASE_URL = "http://127.0.0.1:8000";


/* =========================
   DOM ELEMENTS
========================= */

const apiStatus = document.getElementById("api-status");

const storeCount = document.getElementById("store-count");

const familyCount = document.getElementById("family-count");

const bestModel = document.getElementById("best-model");

const apiKpi = document.getElementById("api-kpi");

const metricsHeader = document.getElementById("metrics-header");

const metricsBody = document.getElementById("metrics-body");

const storesList = document.getElementById("stores-list");

const familiesList = document.getElementById("families-list");

const systemMessage = document.getElementById("system-message");


/* =========================
   API HELPER
========================= */

async function fetchAPI(endpoint) {

    const response = await fetch(`${API_BASE_URL}${endpoint}`);

    if (!response.ok) {

        throw new Error(
            `API request failed: ${response.status}`
        );

    }

    return await response.json();
}


/* =========================
   HEALTH CHECK
========================= */

async function loadHealth() {

    try {

        const data = await fetchAPI("/health");


        apiStatus.textContent = "API Online";

        apiKpi.textContent = "Online";

        systemMessage.textContent =
            `${data.service} is running successfully.`;


    } catch (error) {

        console.error("Health check failed:", error);

        apiStatus.textContent = "API Offline";

        apiKpi.textContent = "Offline";

        systemMessage.textContent =
            "Unable to connect to the FastAPI backend.";

    }

}


/* =========================
   LOAD STORES
========================= */

async function loadStores() {

    try {

        const data = await fetchAPI("/stores");


        storeCount.textContent = data.count;


        storesList.innerHTML = "";


        data.stores.forEach(store => {

            const tag = document.createElement("span");

            tag.className = "tag";

            tag.textContent = `Store ${store}`;

            storesList.appendChild(tag);

        });


    } catch (error) {

        console.error("Failed to load stores:", error);

        storeCount.textContent = "--";

        storesList.textContent =
            "Unable to load stores.";

    }

}


/* =========================
   LOAD PRODUCT FAMILIES
========================= */

async function loadFamilies() {

    try {

        const data = await fetchAPI("/families");


        familyCount.textContent = data.count;


        familiesList.innerHTML = "";


        data.families.forEach(family => {

            const tag = document.createElement("span");

            tag.className = "tag";

            tag.textContent = family;

            familiesList.appendChild(tag);

        });


    } catch (error) {

        console.error("Failed to load families:", error);

        familyCount.textContent = "--";

        familiesList.textContent =
            "Unable to load product families.";

    }

}


/* =========================
   LOAD MODEL METRICS
========================= */

async function loadMetrics() {

    try {

        const data = await fetchAPI("/models");


        if (!data.data || data.data.length === 0) {

            metricsBody.innerHTML =
                `<tr>
                    <td colspan="10">
                        No model data available.
                    </td>
                </tr>`;

            bestModel.textContent = "--";

            return;

        }


        /*
         * Determine table columns
         */

        const columns = Object.keys(data.data[0]);


        metricsHeader.innerHTML = "";


        columns.forEach(column => {

            const th = document.createElement("th");

            th.textContent = formatColumnName(column);

            metricsHeader.appendChild(th);

        });


        /*
         * Populate rows
         */

        metricsBody.innerHTML = "";


        data.data.forEach(row => {

            const tr = document.createElement("tr");


            columns.forEach(column => {

                const td = document.createElement("td");

                td.textContent = formatValue(row[column]);

                tr.appendChild(td);

            });


            metricsBody.appendChild(tr);

        });


        /*
         * Find best model
         */

        const modelColumn = findColumn(
            columns,
            ["model", "model_name", "algorithm"]
        );


        if (modelColumn) {

            bestModel.textContent =
                data.data[0][modelColumn];

        } else {

            bestModel.textContent = "--";

        }


    } catch (error) {

        console.error(
            "Failed to load model metrics:",
            error
        );


        metricsBody.innerHTML =
            `<tr>
                <td colspan="10">
                    Unable to load model metrics.
                </td>
            </tr>`;

        bestModel.textContent = "--";

    }

}


/* =========================
   FORMAT COLUMN NAMES
========================= */

function formatColumnName(column) {

    return column
        .replaceAll("_", " ")
        .replace(/\b\w/g, char => char.toUpperCase());

}


/* =========================
   FORMAT VALUES
========================= */

function formatValue(value) {

    if (value === null || value === undefined) {

        return "-";

    }


    if (typeof value === "number") {

        return Number.isInteger(value)
            ? value.toString()
            : value.toFixed(4);

    }


    return value;

}


/* =========================
   FIND COLUMN
========================= */

function findColumn(columns, possibleNames) {

    return columns.find(column => {

        return possibleNames.includes(
            column.toLowerCase()
        );

    });

}


/* =========================
   INITIALIZE DASHBOARD
========================= */

async function initializeDashboard() {

    console.log(
        "Initializing Retail Demand Forecasting Dashboard..."
    );


    await Promise.all([

        loadHealth(),

        loadStores(),

        loadFamilies(),

        loadMetrics()

    ]);


    console.log(
        "Dashboard initialization complete."
    );

}


/* =========================
   START APPLICATION
========================= */

initializeDashboard();