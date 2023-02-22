function formSwitch(check_class, div_id) {
    let check = document.getElementsByClassName(check_class)
    let masterBox = document.getElementById(div_id);

    if (check[0].checked) {
        masterBox.style.display = "none";

    } else if (check[1].checked || check[2].checked) {
        masterBox.style.display = "block";

    } else {
        masterBox.style.display = "none";
    }
}

function formSwitch_cluster(check_id, div_id) {
    let check = document.getElementById(check_id);
    let clusterBox = document.getElementById(div_id);

    if (check.checked == true) {
        clusterBox.style.display = "block";
    } else {
        clusterBox.style.display = "none";
    }
}

function formSwitch_chemtype(check_id, div_id) {
    let check = document.getElementById(check_id).value;
    let masterBox = document.getElementById(div_id);

    if (check=="mfp" || check=="maccs") {
        masterBox.style.display = "block";
    }
    else {
        masterBox.style.display = "none";
    }

}