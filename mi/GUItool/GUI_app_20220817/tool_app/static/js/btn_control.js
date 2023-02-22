/*
Set button control (add, delete, disable)
*/

//入力フォーム、プルダウンから値を抽出
function extractVals(div) {
    var vals = [];
    div.each(function () {
        vals.push($(this).val());
    });

    return vals;
};

//チェックボックスから値を抽出
function checkboxVals(div) {
    var vals = [];
    div.each(function () {
        if ($(this).prop('checked')) {
            vals.push($(this).val());
        }
        else {
            vals.push('');
        }
    });

    return vals;
};

//処理中にボタンを非活性にする
function btn_disable(){
    $("#btn_submit_preview").prop("disabled", true);
    $("#btn_file_select").prop("disabled", true);
    $("#btn_file_upload").prop("disabled", true);
    $("#btn_submit_register").prop("disabled", true);
    $("#btn_submit_visualization").prop("disabled", true);
    $("#model_building_button").prop("disabled", true);
    $("#btn_submit_condition").prop("disabled", true);
    $('#btn_submit_target').prop("disabled", true);
    $("#btn_submit_mfp").prop("disabled", true);
    $("#btn_submit_top").prop("disabled", true);
    $("#btn_submit_draw").prop("disabled", true);
}

//処理中にボタンを非活性にする
function btn_able(){
    $("#btn_submit_preview").prop("disabled", false);
    $("#btn_file_select").prop("disabled", false);
    $("#btn_file_upload").prop("disabled", false);
    $("#btn_submit_register").prop("disabled", false);
    $("#btn_submit_visualization").prop("disabled", false);
    $("#model_building_button").prop("disabled", false);
    $("#btn_submit_condition").prop("disabled", false);
    $('#btn_submit_target').prop("disabled", false);
    $("#btn_submit_mfp").prop("disabled", false);
    $("#btn_submit_top").prop("disabled", false);
    $("#btn_submit_draw").prop("disabled", false);
}

//追加ボタンの挙動
function btn_add(div_class, cb_with = false){
    let divs = $(div_class).last();

    if (!cb_with){
        divs.clone().find("input:text").val('').end()
        .insertAfter(divs);
    }
    else{
        divs.clone().find("input:text").val('').end()
        .find("input:checkbox").prop('checked', false).end()
        .insertAfter(divs);
    }
}

//追加ボタンと削除ボタンの挙動
function btn_add_del(btn_clone, btn_remove, div_class, cb_with){
    //追加
    btn_clone.click(function() {
        btn_add(div_class, cb_with);

        if($(div_class).length >=2) {
            $(btn_remove).show();
        }
    });
    //削除
    btn_remove.click(function() {
        $(div_class).last().remove();
        
        if($(div_class).length < 2) {
            $(btn_remove).hide();
        }
    });
}

//追加ボタンと削除ボタンの挙動
function btn_add_del_groupsum(btn_clone, btn_remove, div_class, cb_with){
    //追加
    btn_clone.click(function() {
        btn_add(div_class, cb_with);

        if($(div_class).length >=2) {
            $(btn_remove).show();
        }
    });
    //削除
    btn_remove.click(function() {
        $(div_class).last().remove();
        
        if($(div_class).length < 2) {
            $(btn_remove).hide();
        }
    });
}

//パラメータの追加と削除 range
$(function() {
    const btn_clone=$('#btn_clone_range');
    const btn_remove=$('#btn_remove_range');
    $(btn_remove).hide();
    const div_class = ".div_range";
    
    btn_add_del(btn_clone, btn_remove, div_class);
}); 

//パラメータの追加と削除 fixed
$(function() {
    const btn_clone = $('#btn_clone_fixed');
    const btn_remove = $('#btn_remove_fixed');
    $(btn_remove).hide();
    const div_class = ".div_fixed";
    
    btn_add_del(btn_clone, btn_remove, div_class);
}); 

//パラメータの追加と削除 total
$(function() {
    const btn_clone = $('#btn_clone_total');
    const btn_remove = $('#btn_remove_total');
    $(btn_remove).hide();
    const div_class = ".div_total";
    
    btn_add_del(btn_clone, btn_remove, div_class, true);
}); 

//パラメータの追加と削除 combination
$(function() {
    const btn_clone = $('#btn_clone_combination');
    const btn_remove = $('#btn_remove_combination');
    $(btn_remove).hide();
    const div_class = ".div_combination";
    
    btn_add_del(btn_clone, btn_remove, div_class, true);
}); 

//パラメータの追加と削除 ratio
$(function() {
    const btn_clone = $('#btn_clone_ratio');
    const btn_remove = $('#btn_remove_ratio');
    $(btn_remove).hide();
    const div_class = ".div_ratio";
    
    btn_add_del(btn_clone, btn_remove, div_class);
});


//パラメータの追加と削除 group
function group_add(group_id){
    let group_id_name = "#"+group_id

    const btn_clone = $(group_id_name + " .btn_clone_group");
    const btn_remove = $(group_id_name + " .btn_remove_group");
    $(btn_remove).hide();
    const div_class = group_id_name + " .div_group";
    btn_add_del(btn_clone, btn_remove, div_class, true);
}


$(function() {
    group_add("id_groupsum_0");
    group_add("id_groupsum_1");
    group_add("id_groupsum_2");
    group_add("id_groupsum_3");
    group_add("id_groupsum_4");

});


let group_i = 0
//パラメータの追加と削除 groupsum
$(function() {
    const btn_clone = $('#btn_clone_groupsum');
    const btn_remove = $('#btn_remove_groupsum');
    $(btn_remove).hide();
    const div_class = ".div_groupsum";

    //追加
    btn_clone.click(function() {
        //let divs = $(div_class).last();
        group_i = group_i+1;
        let div_1 = document.getElementById("id_groupsum_1");
        let div_2 = document.getElementById("id_groupsum_2");
        let div_3 = document.getElementById("id_groupsum_3");
        let div_4 = document.getElementById("id_groupsum_4");

        if (group_i >= 5) {
            group_i = 4
        }
        if (group_i == 4) {
            $(btn_clone).hide();
        }
        if (group_i > 0) {
            $(btn_remove).show();
        }

        if (group_i >= 1) {
            div_1.style.display = "block";
        }
        if (group_i>=2) {
            div_2.style.display = "block";        
        }
        if (group_i>=3) {
            div_3.style.display = "block";        
        }
        if (group_i>=4) {
            div_4.style.display = "block";        
        }


        /*
        divs.clone().find("input:text").val('').end()
        .find("input:checkbox").prop('checked', false).end()
        .attr("id", "id_groupsum_"+group_i)
        .insertAfter(divs);

        if($(div_class).length >=2) {
            $(btn_remove).show();
        }
        */
    });
    
    //削除
    btn_remove.click(function() {
        //$(div_class).last().remove();
        group_i = group_i-1;
        let div_1 = document.getElementById("id_groupsum_1");
        let div_2 = document.getElementById("id_groupsum_2");
        let div_3 = document.getElementById("id_groupsum_3");
        let div_4 = document.getElementById("id_groupsum_4");

        if (group_i<=0)
        {
            group_i = 0
        }
        if (group_i == 0) {
            $(btn_remove).hide();
        }
        if (group_i > 0) {
            $(btn_clone).show();
        }

        if (group_i < 1) {
            div_1.style.display = "none";
        }
        if (group_i < 2) {
            div_2.style.display = "none";
        }
        if (group_i < 3) {
            div_3.style.display = "none";
        }
        if (group_i < 4) {
            div_4.style.display = "none";
        }


        /*
        if($(div_class).length < 2) {
            $(btn_remove).hide();
        }
        */
    });
}); 

//パラメータの追加と削除 target
$(function() {
    const btn_clone = $('#btn_clone_target');
    const btn_remove = $('#btn_remove_target');
    $(btn_remove).hide();
    const div_class = ".div_target";
    
    btn_add_del(btn_clone, btn_remove, div_class);
});

// 刻み値の設定 step
$(function() {
    const btn_clone = $('#btn_clone_step');
    const btn_remove = $('#btn_remove_step');
    $(btn_remove).hide();
    const div_class = ".div_step";
    
    btn_add_del(btn_clone, btn_remove, div_class);
});