/* =========================================================
   RETAIL DEMAND FORECASTING DASHBOARD
========================================================= */

const API_BASE = "http://127.0.0.1:8000";

let inventoryData = [];
let forecastData = [];


/* =========================================================
   HELPERS
========================================================= */

function formatNumber(value, decimals = 2) {

    const number = Number(value);

    if (!Number.isFinite(number)) {
        return "--";
    }

    return number.toLocaleString(
        "en-US",
        {
            minimumFractionDigits: decimals,
            maximumFractionDigits: decimals
        }
    );
}


function escapeHtml(value) {

    if (value === null || value === undefined) {
        return "";
    }

    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}


async function fetchJSON(endpoint) {

    const response = await fetch(
        `${API_BASE}${endpoint}`
    );

    if (!response.ok) {

        throw new Error(
            `API request failed: ${response.status}`
        );
    }

    return await response.json();
}


/* =========================================================
   API STATUS
========================================================= */

async function checkAPI() {

    const statusText =
        document.getElementById("apiStatus");

    const statusKpi =
        document.getElementById("apiKpi");

    const dot =
        document.getElementById("statusDot");

    try {

        await fetchJSON("/health");

        statusText.textContent = "Online";

        statusKpi.textContent = "Online";

        dot.classList.remove("offline");

        dot.classList.add("online");

    } catch (error) {

        statusText.textContent = "Offline";

        statusKpi.textContent = "Offline";

        dot.classList.remove("online");

        dot.classList.add("offline");

        console.error(
            "API health check failed:",
            error
        );
    }
}


/* =========================================================
   OVERVIEW
========================================================= */

async function loadOverview() {

    try {

        const [
            stores,
            families,
            models
        ] = await Promise.all([
            fetchJSON("/stores"),
            fetchJSON("/families"),
            fetchJSON("/models")
        ]);


        document.getElementById(
            "totalStores"
        ).textContent = stores.count;


        document.getElementById(
            "totalFamilies"
        ).textContent = families.count;


        populateSelect(
            "forecastStore",
            stores.stores,
            "All Stores"
        );

        populateSelect(
            "inventoryStore",
            stores.stores,
            "All Stores"
        );


        populateSelect(
            "forecastFamily",
            families.families,
            "All Families"
        );

        populateSelect(
            "inventoryFamily",
            families.families,
            "All Families"
        );


        renderModels(models.data);

    } catch (error) {

        console.error(
            "Overview loading failed:",
            error
        );
    }
}


/* =========================================================
   SELECT POPULATION
========================================================= */

function populateSelect(
    elementId,
    values,
    defaultText
) {

    const select =
        document.getElementById(elementId);

    if (!select) {
        return;
    }

    select.innerHTML = "";

    const defaultOption =
        document.createElement("option");

    defaultOption.value = "";

    defaultOption.textContent = defaultText;

    select.appendChild(defaultOption);


    values.forEach(value => {

        const option =
            document.createElement("option");

        option.value = value;

        option.textContent = value;

        select.appendChild(option);

    });
}


/* =========================================================
   MODEL PERFORMANCE
========================================================= */

function renderModels(models) {

    const body =
        document.getElementById(
            "modelTableBody"
        );

    if (!models || models.length === 0) {

        body.innerHTML = `
            <tr>
                <td colspan="5" class="empty">
                    No model performance data available.
                </td>
            </tr>
        `;

        return;
    }


    body.innerHTML = models.map(model => {

        const modelName =
            model.model ||
            model.Model ||
            model.name ||
            model.model_name ||
            "--";


        const mae =
            model.MAE ??
            model.mae ??
            "--";


        const rmse =
            model.RMSE ??
            model.rmse ??
            "--";


        const rmsle =
            model.RMSLE ??
            model.rmsle ??
            "--";


        const wape =
            model.WAPE ??
            model.wape ??
            "--";


        return `
            <tr>
                <td>
                    <strong>
                        ${escapeHtml(modelName)}
                    </strong>
                </td>

                <td>${formatNumber(mae)}</td>

                <td>${formatNumber(rmse)}</td>

                <td>${formatNumber(rmsle)}</td>

                <td>${formatNumber(wape)}%</td>
            </tr>
        `;

    }).join("");


    /*
       Find the model with the lowest MAE.
    */

    let bestModel = null;

    let bestMAE = Infinity;


    models.forEach(model => {

        const mae =
            Number(
                model.MAE ??
                model.mae
            );

        if (
            Number.isFinite(mae) &&
            mae < bestMAE
        ) {

            bestMAE = mae;

            bestModel =
                model.model ||
                model.Model ||
                model.name ||
                model.model_name;

        }

    });


    if (bestModel) {

        document.getElementById(
            "bestModel"
        ).textContent = bestModel;

    }

}


/* =========================================================
   FORECAST
========================================================= */

async function loadForecast() {

    const body =
        document.getElementById(
            "forecastTableBody"
        );

    body.innerHTML = `
        <tr>
            <td colspan="6" class="loading">
                Loading forecasts...
            </td>
        </tr>
    `;


    const store =
        document.getElementById(
            "forecastStore"
        ).value;


    const family =
        document.getElementById(
            "forecastFamily"
        ).value;


    const dateFrom =
        document.getElementById(
            "dateFrom"
        ).value;


    const dateTo =
        document.getElementById(
            "dateTo"
        ).value;


    const params =
        new URLSearchParams();


    params.set("limit", "50000");


    if (store) {
        params.set("store", store);
    }

    if (family) {
        params.set("family", family);
    }

    if (dateFrom) {
        params.set("date_from", dateFrom);
    }

    if (dateTo) {
        params.set("date_to", dateTo);
    }


    try {

        const result =
            await fetchJSON(
                `/forecast?${params.toString()}`
            );


        forecastData =
            result.data || [];


        renderForecast(
            forecastData
        );


    } catch (error) {

        console.error(
            "Forecast loading failed:",
            error
        );


        body.innerHTML = `
            <tr>
                <td colspan="6" class="error">
                    Unable to load forecast data.
                    Make sure the FastAPI server is running.
                </td>
            </tr>
        `;
    }
}


/* =========================================================
   RENDER FORECAST
========================================================= */

function renderForecast(data) {

    const body =
        document.getElementById(
            "forecastTableBody"
        );


    if (!data || data.length === 0) {

        body.innerHTML = `
            <tr>
                <td colspan="6" class="empty">
                    No forecast records found.
                </td>
            </tr>
        `;

        return;
    }


    /*
       Show the first 100 records in the dashboard.
       The API still loads up to 50,000 records.
    */

    const rows =
        data.slice(0, 100);


    body.innerHTML =
        rows.map(row => {

            const actual =
                Number(
                    row.actual_sales ??
                    row.sales ??
                    row.actual ??
                    0
                );


            const predicted =
                Number(
                    row.predicted_sales ??
                    row.prediction ??
                    row.predicted ??
                    0
                );


            const error =
                predicted - actual;


            return `
                <tr>

                    <td>
                        ${escapeHtml(row.date ?? "--")}
                    </td>

                    <td>
                        ${escapeHtml(row.store_nbr ?? "--")}
                    </td>

                    <td>
                        ${escapeHtml(row.family ?? "--")}
                    </td>

                    <td>
                        ${formatNumber(actual)}
                    </td>

                    <td>
                        ${formatNumber(predicted)}
                    </td>

                    <td>
                        ${formatNumber(error)}
                    </td>

                </tr>
            `;

        }).join("");

}


/* =========================================================
   INVENTORY
========================================================= */

async function loadInventory() {

    const body =
        document.getElementById(
            "inventoryTableBody"
        );


    body.innerHTML = `
        <tr>
            <td colspan="9" class="loading">
                Loading inventory recommendations...
            </td>
        </tr>
    `;


    try {

        /*
           IMPORTANT:
           We load the complete inventory dataset from
           the FastAPI endpoint.

           The backend currently has 1,782 recommendations.
        */

        const result =
            await fetchJSON(
                "/inventory?limit=50000"
            );


        inventoryData =
            result.data || [];


        console.log(
            "Inventory data loaded:",
            inventoryData.length
        );


        updatePrioritySummary(
            inventoryData
        );


        renderInventory();


    } catch (error) {

        console.error(
            "Inventory loading failed:",
            error
        );


        body.innerHTML = `
            <tr>
                <td colspan="9" class="error">
                    Unable to load inventory recommendations.
                    Make sure FastAPI is running on port 8000.
                </td>
            </tr>
        `;


        document.getElementById(
            "inventoryMessage"
        ).textContent =
            error.message;
    }
}


/* =========================================================
   PRIORITY SUMMARY
========================================================= */

function updatePrioritySummary(data) {

    let high = 0;

    let medium = 0;

    let low = 0;


    data.forEach(row => {

        const priority =
            String(
                row.inventory_priority || ""
            )
            .trim()
            .toUpperCase();


        if (priority === "HIGH PRIORITY") {

            high++;

        } else if (
            priority === "MEDIUM PRIORITY"
        ) {

            medium++;

        } else if (
            priority === "LOW PRIORITY"
        ) {

            low++;
        }

    });


    document.getElementById(
        "highPriorityCount"
    ).textContent = high;


    document.getElementById(
        "mediumPriorityCount"
    ).textContent = medium;


    document.getElementById(
        "lowPriorityCount"
    ).textContent = low;


    document.getElementById(
        "totalInventoryCount"
    ).textContent = data.length;
}


/* =========================================================
   INVENTORY FILTER
========================================================= */

function renderInventory() {

    const body =
        document.getElementById(
            "inventoryTableBody"
        );


    const priority =
        document.getElementById(
            "inventoryPriority"
        ).value;


    const store =
        document.getElementById(
            "inventoryStore"
        ).value;


    const family =
        document.getElementById(
            "inventoryFamily"
        ).value;


    let filtered =
        inventoryData.filter(row => {

            const rowPriority =
                String(
                    row.inventory_priority || ""
                )
                .trim()
                .toUpperCase();


            const rowStore =
                String(
                    row.store_nbr ?? ""
                );


            const rowFamily =
                String(
                    row.family ?? ""
                );


            const priorityMatch =
                priority === "ALL" ||
                rowPriority === priority;


            const storeMatch =
                !store ||
                rowStore === String(store);


            const familyMatch =
                !family ||
                rowFamily === family;


            return (
                priorityMatch &&
                storeMatch &&
                familyMatch
            );

        });


    /*
       Sort by priority first.
    */

    const priorityOrder = {
        "HIGH PRIORITY": 1,
        "MEDIUM PRIORITY": 2,
        "LOW PRIORITY": 3
    };


    filtered.sort(
        (a, b) => {

            const aPriority =
                String(
                    a.inventory_priority || ""
                )
                .trim()
                .toUpperCase();


            const bPriority =
                String(
                    b.inventory_priority || ""
                )
                .trim()
                .toUpperCase();


            return (
                (priorityOrder[aPriority] || 99) -
                (priorityOrder[bPriority] || 99)
            );

        }
    );


    if (filtered.length === 0) {

        body.innerHTML = `
            <tr>
                <td colspan="9" class="empty">
                    No inventory recommendations
                    match the selected filters.
                </td>
            </tr>
        `;


        document.getElementById(
            "inventoryMessage"
        ).textContent =
            "0 recommendations found.";

        return;
    }


    /*
       Render maximum 500 rows in browser.
    */

    const rows =
        filtered.slice(0, 500);


    body.innerHTML =
        rows.map(row => {

            const priority =
                String(
                    row.inventory_priority || "UNKNOWN"
                )
                .trim()
                .toUpperCase();


            let badgeClass =
                "priority-unknown";


            if (
                priority === "HIGH PRIORITY"
            ) {

                badgeClass =
                    "priority-high";

            } else if (
                priority === "MEDIUM PRIORITY"
            ) {

                badgeClass =
                    "priority-medium";

            } else if (
                priority === "LOW PRIORITY"
            ) {

                badgeClass =
                    "priority-low";
            }


            return `
                <tr>

                    <td>
                        <span
                            class="priority-badge ${badgeClass}"
                        >
                            ${escapeHtml(priority)}
                        </span>
                    </td>


                    <td>
                        ${escapeHtml(
                            row.store_nbr ?? "--"
                        )}
                    </td>


                    <td>
                        ${escapeHtml(
                            row.family ?? "--"
                        )}
                    </td>


                    <td>
                        ${escapeHtml(
                            row.city ?? "--"
                        )}
                    </td>


                    <td>
                        ${formatNumber(
                            row.forecast_7_day
                        )}
                    </td>


                    <td>
                        ${formatNumber(
                            row.average_daily_demand
                        )}
                    </td>


                    <td>
                        ${formatNumber(
                            row.safety_stock
                        )}
                    </td>


                    <td>
                        ${formatNumber(
                            row.reorder_point
                        )}
                    </td>


                    <td>
                        <strong>
                            ${formatNumber(
                                row.recommended_stock
                            )}
                        </strong>
                    </td>

                </tr>
            `;

        }).join("");


    document.getElementById(
        "inventoryMessage"
    ).textContent =
        `Showing ${rows.length} of ${filtered.length} recommendations.`;

}


/* =========================================================
   EVENT LISTENERS
========================================================= */

function setupEventListeners() {


    document.getElementById(
        "forecastButton"
    ).addEventListener(
        "click",
        loadForecast
    );


    document.getElementById(
        "inventoryFilterButton"
    ).addEventListener(
        "click",
        renderInventory
    );


    document.getElementById(
        "inventoryRefreshButton"
    ).addEventListener(
        "click",
        loadInventory
    );


    /*
       Optional: changing priority immediately updates table.
    */

    document.getElementById(
        "inventoryPriority"
    ).addEventListener(
        "change",
        renderInventory
    );

}


/* =========================================================
   INITIALIZE DASHBOARD
========================================================= */

async function initDashboard() {

    console.log(
        "Initializing Retail Demand Forecasting Dashboard..."
    );


    setupEventListeners();


    /*
       Load all dashboard components.
    */

    await Promise.all([
        checkAPI(),
        loadOverview(),
        loadForecast(),
        loadInventory()
    ]);


    console.log(
        "Dashboard initialization complete."
    );
}


/* =========================================================
   START
========================================================= */

window.addEventListener(
    "DOMContentLoaded",
    initDashboard
);